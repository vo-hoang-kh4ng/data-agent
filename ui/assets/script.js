() => {
    // 1. Static buttons toggle logic using event delegation to be completely immune to DOM updates
    document.addEventListener('click', function (event) {
        // --- Handle 'Edit Code' button click ---
        const ed_btn = event.target.closest('.ed_btn');
        if (ed_btn) {
            const ed_group = document.querySelector('.ed');
            if (ed_group) {
                ed_group.style.display = ed_group.style.display === 'none' ? 'block' : 'none';
            }
        }

        // --- Handle 'DataFrame' button click ---
        const df_btn = event.target.closest('.df_btn');
        if (df_btn) {
            const df_group = document.querySelector('.df');
            if (df_group) {
                df_group.style.display = df_group.style.display === 'none' ? 'block' : 'none';
            }
        }

        // --- Handle dynamically added 'suggestion-btn' buttons ---
        const suggestion_btn = event.target.closest('.suggestion-btn');
        if (suggestion_btn) {
            console.log("Suggestion button clicked: " + suggestion_btn.textContent);
            // Bulletproof selectors to locate the text area or input field in the page
            const textarea = document.querySelector('#chatbot_input textarea') || 
                             document.querySelector('#chatbot_input input') || 
                             document.querySelector('textarea[placeholder*="message"]') || 
                             document.querySelector('input[placeholder*="message"]') ||
                             document.querySelector('textarea') ||
                             document.querySelector('input[type="text"]');
            
            if (textarea) {
                textarea.value = suggestion_btn.textContent.trim();
                // Crucial: Dispatch 'input' event so Gradio/Svelte state detects the change
                textarea.dispatchEvent(new Event("input", { bubbles: true }));
                textarea.focus();
            } else {
                console.error("Could not locate the Gradio chat input element!");
            }
        }
    });
}
