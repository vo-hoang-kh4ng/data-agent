import openai
from triadic_dgm.prompts.prompts import PROGRAMMER_PROMPT
from triadic_dgm.knowledge.knw_in import retrieval_knowledge
import os
import re
import time
import traceback
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#: Máy chủ tự nói ra giới hạn thật khi từ chối. Bắt lấy câu đó thay vì tin một hằng số.
_CONTEXT_LIMIT_RE = re.compile(r"maximum context length is\s+(\d+)\s*tokens", re.IGNORECASE)


def parse_context_limit(message):
    """Số token tối đa mà máy chủ vừa nói ra trong lời từ chối, hoặc None.

    Hằng số MODEL_CONTEXT_LIMIT đã sai hai lần theo hai hướng ngược nhau: 30000 làm mọi
    script sinh ra bị cắt giữa dòng, sửa lên 62000 bằng cách hỏi người, rồi phía phục vụ
    tụt xuống 32000 và cả run chết vì vượt đúng MỘT token. Một con số mô tả cái máy mà code
    không nhìn thấy được thì sẽ còn hỏng nữa, và mỗi lần hỏng là hỏng vào giữa phiên làm việc
    của người dùng.

    Chỉ nhận diện đúng lời từ chối vì độ dài. Học một giới hạn từ lỗi 504 sẽ bóp ngân sách
    lại mà không vì lý do gì.

    Args:
        message: Nội dung lỗi trả về (có thể là None).

    Returns:
        Giới hạn theo token, hoặc None nếu lỗi không nói gì về độ dài ngữ cảnh.
    """
    if not message:
        return None
    found = _CONTEXT_LIMIT_RE.search(str(message))
    return int(found.group(1)) if found else None

# Qwen3.5 (hosted qua proxy proxy.onebot.meobeo.ai) hỗ trợ chế độ "thinking" — sinh 1 đoạn suy luận
# nội bộ dài trước khi trả lời thật, làm request lâu hơn nhiều và dễ chạm timeout của gateway (504,
# ĐÃ XẢY RA NHIỀU LẦN trên live run, request treo ~90s trước khi gateway trả 504). Tắt hẳn để giảm
# thời gian sinh — cùng pattern đã dùng cho đúng model/proxy này ở
# triadic_dgm/benchmark/implementations/qwen_llm.py (comment gốc: "proxy thường xuyên rớt").
_DISABLE_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


class Programmer:

    # Chỉ là ĐIỂM XUẤT PHÁT, không phải sự thật. Con số này đã sai hai lần theo hai hướng
    # ngược nhau: 30000 làm mọi script sinh ra bị cắt giữa dòng (cùng một SyntaxError, cùng
    # một chỗ cắt, ở mọi lần thử lại — thử lại không chữa được giới hạn token); sửa lên
    # 62000 sau khi hỏi người dùng; rồi phía phục vụ tụt xuống 32000 và cả run chết vì vượt
    # đúng MỘT token. Nguồn sự thật, theo thứ tự: biến môi trường LLM_CONTEXT_LIMIT nếu người
    # vận hành khai, còn không thì chính lời từ chối của máy chủ — xem parse_context_limit().
    MODEL_CONTEXT_LIMIT = 62000
    OUTPUT_TOKENS_FLOOR = 1500
    OUTPUT_TOKENS_CEILING = 20000
    CONTEXT_SAFETY_MARGIN = 1000
    CHARS_PER_TOKEN_ESTIMATE = 3  # conservative for mixed Vietnamese/English/code content

    def __init__(self, api_key, model="gpt-4o-mini", base_url=None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.messages = []
        self.function_repository = {}
        self.last_snaps = None
        # Người vận hành biết deployment thì khai báo thẳng, không phải gây ra một lần lỗi
        # mới cấu hình được. Không khai thì dùng mặc định và HỌC lại từ lời từ chối đầu tiên.
        self.context_limit = self._configured_context_limit()

    @classmethod
    def _configured_context_limit(cls) -> int:
        declared = os.getenv("LLM_CONTEXT_LIMIT", "").strip()
        try:
            value = int(declared)
        except ValueError:
            return cls.MODEL_CONTEXT_LIMIT
        return value if value > 0 else cls.MODEL_CONTEXT_LIMIT

    def _history_char_budget(self) -> int:
        """Số ký tự hội thoại được phép giữ, suy ra từ cửa sổ ĐANG dùng.

        Trước đây là 120.000 gõ cứng, chọn khi cửa sổ được tin là 62.000 token. Trên máy chủ
        32.000 thì ngưỡng ấy không bao giờ chạm tới trước khi cửa sổ đã đầy: chỉ một vòng sửa
        lỗi là đã nối thêm nguyên một script sinh ra, và _compute_max_tokens() sau đó chỉ trả
        về đúng cái sàn — đó là lý do script quay lại bị cắt giữa dòng. Cùng công thức này
        cho ra lại xấp xỉ 120.000 ở giới hạn cũ.
        """
        room = self.context_limit - self.OUTPUT_TOKENS_CEILING - self.CONTEXT_SAFETY_MARGIN
        return max(self.OUTPUT_TOKENS_FLOOR, room) * self.CHARS_PER_TOKEN_ESTIMATE

    def _trim_history(self) -> None:
        """Bỏ bớt lượt cũ cho tới khi hội thoại vừa cửa sổ.

        Giữ nguyên System Prompt (0) và Task/Domain Knowledge ban đầu (1) — cắt qua hai cái
        đó là để model không còn chỉ dẫn lẫn đề bài.
        """
        budget = self._history_char_budget()
        while (sum(len(str(m.get("content", ""))) for m in self.messages) > budget
               and len(self.messages) > 2):
            self.messages.pop(2)

    def _learn_context_limit(self, error) -> bool:
        """Nhận giới hạn thật từ lời từ chối. True nếu ngân sách vừa đổi.

        Thử lại y nguyên request cũ thì không bao giờ qua được — tràn ngữ cảnh không có gì
        là tạm thời. Đường streaming trước đây thử lại đúng ba lần kèm backoff, tức là chỉ
        có thể hỏng ba lần rồi bỏ cuộc.
        """
        stated = parse_context_limit(str(error))
        if stated is None or stated >= self.context_limit:
            return False
        print(f"[PROGRAMMER] máy chủ báo cửa sổ ngữ cảnh thật là {stated:,} token "
              f"(đang dùng {self.context_limit:,}) — nhận lại và thử lại một lần")
        self.context_limit = stated
        return True

    def add_functions(self, function_lib: dict) -> None:
        self.function_repository = function_lib

    def _compute_max_tokens(self) -> int:
        prompt_chars = sum(len(str(m.get("content", ""))) for m in self.messages)
        estimated_prompt_tokens = prompt_chars // self.CHARS_PER_TOKEN_ESTIMATE
        available = self.context_limit - estimated_prompt_tokens - self.CONTEXT_SAFETY_MARGIN
        budget = max(self.OUTPUT_TOKENS_FLOOR, min(self.OUTPUT_TOKENS_CEILING, available))
        # Sàn OUTPUT_TOKENS_FLOOR có lý do: xin quá ít thì script sinh ra bị cắt giữa dòng.
        # Nhưng khi prompt đã gần lấp kín cửa sổ thì cái sàn ấy đẩy tổng vượt giới hạn, và
        # một request bị từ chối thì không sinh ra được ký tự nào. Trả lời cụt vẫn hơn.
        return max(1, min(budget, self.context_limit - estimated_prompt_tokens))

    def _call_chat_model(self, functions=None, include_functions=False, retrieval=False):
        if retrieval:
            snaps = retrieval_knowledge(self.messages[-1]["content"])
            if snaps:
                self.last_snaps = snaps
                self.messages[-1]["content"] += snaps
            else:
                self.last_snaps = None

        # Giữ đủ chỗ cho một lượt sinh trọn vẹn sau khi vài vòng sửa lỗi đã chất lịch sử lên
        # (mỗi lần hỏng nối thêm nguyên script vừa sinh). Ngân sách suy ra từ cửa sổ ĐANG
        # dùng — xem _history_char_budget().
        self._trim_history()

        params = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self._compute_max_tokens(),
            "extra_body": _DISABLE_THINKING_EXTRA_BODY,
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        # Một lần sửa cho mỗi lệnh gọi: lần đầu có thể xin quá tay vì hằng số đã cũ, lần hai
        # xin theo đúng con số máy chủ vừa nói. Máy chủ từ chối cả cái nó vừa yêu cầu là lỗi
        # thật, không phải thứ để đuổi theo.
        for corrected in (False, True):
            try:
                response = self.client.chat.completions.create(**params)
                usage = response.usage
                print(f"======Prompt Tokens: {usage.prompt_tokens}======Completion Tokens: {usage.completion_tokens}=======Total Tokens: {usage.total_tokens}")
                return response
            except Exception as e:
                print(f"Error calling chat model: {e}")
                if corrected or not self._learn_context_limit(e):
                    return None
                params["max_tokens"] = self._compute_max_tokens()
        return None

    def _call_chat_model_streaming(self, functions=None, include_functions=False, retrieval=False, kernel=None):
        temp = self.messages[-1]["content"]
        if retrieval:
            snaps = retrieval_knowledge(self.messages[-1]["content"], kernel=kernel)
            if snaps:
                for chunk in snaps:
                    yield chunk
                self.last_snaps = snaps
                self.messages[-1]["content"] += snaps
            else:
                self.last_snaps = None

        # Giữ đủ chỗ cho một lượt sinh trọn vẹn sau khi vài vòng sửa lỗi đã chất lịch sử lên
        # (mỗi lần hỏng nối thêm nguyên script vừa sinh). Ngân sách suy ra từ cửa sổ ĐANG
        # dùng — xem _history_char_budget().
        self._trim_history()

        params = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "max_tokens": self._compute_max_tokens(),
            "extra_body": _DISABLE_THINKING_EXTRA_BODY,
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        # Retry với backoff — đây là lệnh gọi LLM NẶNG NHẤT và chạy ĐẦU TIÊN trong cả pipeline (sinh
        # toàn bộ code K-Means/business-rules/JSON), trước đây KHÔNG có retry nào cả: 1 lần gateway
        # timeout (504, cùng loại lỗi đã gặp ở report_generator.py) là mất trắng, không có gì để
        # fallback (khác narrative LLM call — cái đó còn rơi về bản deterministic). CHỈ retry khi
        # CHƯA yield được chunk nào (stream fail ngay từ đầu, an toàn để thử lại từ đầu) — nếu đã
        # stream ra 1 phần nội dung rồi mới fail thì KHÔNG retry (sẽ bị lặp nội dung trong chat).
        max_attempts = 3
        for attempt in range(max_attempts):
            yielded_any = False
            try:
                stream = self.client.chat.completions.create(**params)
                self.messages[-1]["content"] = temp
                for chunk in stream:
                    if (hasattr(chunk, 'choices') and
                            chunk.choices and
                            len(chunk.choices) > 0 and
                            chunk.choices[0].delta.content is not None):
                        chunk_message = chunk.choices[0].delta.content
                        yielded_any = True
                        yield chunk_message
                return
            except Exception as e:
                print(f"Error calling chat model (attempt {attempt + 1}/{max_attempts}): {e}")
                traceback.print_exc()
                # Tràn ngữ cảnh không phải lỗi tạm thời: thử lại y nguyên thì hỏng y nguyên,
                # và vòng backoff này chỉ có thể hỏng ba lần rồi bỏ cuộc — đúng cái đã xảy ra.
                # Nhận lại giới hạn rồi thử tiếp ngay, không chờ.
                if not yielded_any and self._learn_context_limit(e):
                    params["max_tokens"] = self._compute_max_tokens()
                    continue
                if yielded_any or attempt == max_attempts - 1:
                    yield f"\n\n[LLM ERROR: {e}]\n\n"
                    return
                time.sleep(2 * (attempt + 1))  # 2s, rồi 4s trước lần thử tiếp theo

    def clear(self):
        # IMPORTANT — do NOT .format() PROGRAMMER_PROMPT here (tried it, reverted). The prompt's
        # embedded reference pipeline uses {{double-brace}} escapes so LAMBDA.py's one-time
        # .format(working_path=...) at init produces valid Python. But that one-time formatted
        # copy is immediately overwritten: clear() runs on every new chat (chat.py) and every
        # convergence-loop iteration, and the system has been tuned against the RAW (doubled-
        # brace) prompt ever since — the LLM can't copy the malformed reference code verbatim,
        # so it IMPROVISES working clustering code. Confirmed live: 358 convergence runs on
        # 07-14 with this raw prompt were 100% healthy; switching clear() to .format() (valid
        # braces → LLM copies the canonical pipeline verbatim, incl. its fragile Stage-2 gates)
        # made 100% of runs hard-stop to "Clustering Failed". The doubled-brace state is the
        # battle-tested one — keep it.
        self.messages = [
            {
                "role": "system",
                "content": PROGRAMMER_PROMPT
            }
        ]
        self.function_repository = {}
