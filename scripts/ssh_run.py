import sys
import paramiko
import os

command = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""

try:
    # 1. Connect to gateway
    jumpbox = paramiko.SSHClient()
    jumpbox.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jumpbox.connect(
        hostname=os.getenv('GATEWAY_HOST', '210.245.105.13'),
        port=2702,
        username=os.getenv('GATEWAY_USER', 'fallback_user')
    )
    
    # 2. Open a channel from gateway to the target
    target_host = os.getenv('TARGET_HOST', '172.27.54.58')
    dest_addr = (target_host, 2702)
    local_addr = ('127.0.0.1', 2702)
    jumpbox_transport = jumpbox.get_transport()
    channel = jumpbox_transport.open_channel("direct-tcpip", dest_addr, local_addr)
    
    # 3. Connect to target using the channel
    target = paramiko.SSHClient()
    target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    target.connect(
        hostname=target_host,
        port=2702,
        username=os.getenv('TARGET_USER', 'fallback_user'),
        password=os.getenv('SSH_PASSWORD', 'fallback_if_needed'),
        sock=channel
    )
    
    if command == "transfer":
        sftp = target.open_sftp()
        # Transfer .env and config.yaml
        local_dir = r"d:\kh4ng\Data_agent"
        target_user = os.getenv('TARGET_USER', 'fallback_user')
        remote_dir = f"/home/{target_user}/data-agent"

        
        # Ensure remote directory exists
        try:
            target.exec_command(f"mkdir -p {remote_dir}")
        except:
            pass
            
        print("Transferring files...")
        for f in [".env", "config.yaml"]:
            local_path = os.path.join(local_dir, f)
            remote_path = f"{remote_dir}/{f}"
            if os.path.exists(local_path):
                sftp.put(local_path, remote_path)
                print(f"Transferred {f}")
        sftp.close()
    elif command:
        print(f"Running: {command}")
        stdin, stdout, stderr = target.exec_command(command)
        out = stdout.read().decode()
        err = stderr.read().decode()
        if out: print(out)
        if err: print("STDERR:", err)
        
    target.close()
    jumpbox.close()
    
except Exception as e:
    print("Error:", e)
