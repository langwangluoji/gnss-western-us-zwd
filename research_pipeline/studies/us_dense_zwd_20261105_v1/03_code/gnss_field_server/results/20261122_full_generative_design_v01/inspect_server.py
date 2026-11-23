"""Read-only own-job scheduling inspection, with transient authentication."""
from pathlib import Path
import sys,socket,getpass,json
sys.path.insert(0,'C:/Users/user/AppData/Local/Temp/gnss_full_generative_v01/ssh_lib')
import paramiko
root=Path(__file__).resolve().parent
submission=json.loads((root/'submission_verified.json').read_text(encoding='utf8'))['result']
jobs=[submission[k] for k in ('audit_job_id','preflight_job_id','handoff_job_id')]
if any(not job.isdigit() for job in jobs):raise ValueError('Invalid registered job id')
client=paramiko.SSHClient();client.load_system_host_keys(str(Path.home()/'.ssh/known_hosts'))
secret=getpass.getpass('Tongji password (read-only query, not stored): ')
connection=socket.create_connection(('logini.tongji.edu.cn',10022),timeout=8);connection.settimeout(None)
client.connect('logini.tongji.edu.cn',port=10022,username='u23114',password=secret,sock=connection,
               allow_agent=False,look_for_keys=False,timeout=30,banner_timeout=30,auth_timeout=30)
secret=None
commands={'queue':'squeue -h -j '+','.join(jobs)+' -o "%i|%P|%T|%R|%S"',
          'gpu_job':'scontrol show job '+jobs[1],
          'partition':'sinfo -h -p A800 -o "%P|%a|%D|%G|%t"'}
result={}
try:
    for label,command in commands.items():
        _,out,err=client.exec_command(command,timeout=30)
        result[label]=dict(stdout=out.read().decode('utf8','replace')[-12000:],stderr=err.read().decode('utf8','replace')[-2000:],exit_code=out.channel.recv_exit_status())
    print(json.dumps(result,ensure_ascii=False),flush=True)
    (root/'scheduler_inspection.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf8')
finally:client.close()
