import subprocess
import time
import os

# ================= 설정 부분 =================
TARGET_GPU_ID = 0               # 사용할 GPU 번호 (예: 0번 GPU)
REQUIRED_FREE_VRAM = 20000      # 필요한 여유 메모리 (MB 단위, 예: 20GB가 필요하면 20000)
CHECK_INTERVAL = 30             # 확인 주기 (초 단위, 30초마다 확인)

# 실행할 명령어 (앞서 수정한 인퍼런스 코드를 실행하도록 세팅)
COMMAND = ["python", "src/3_test_inference.py"]
# =============================================

def get_free_vram(gpu_id):
    """지정된 GPU의 현재 여유 메모리(MB)를 가져옵니다."""
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_id}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, 
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except Exception as e:
        print(f"⚠️ nvidia-smi 실행 오류: {e}")
        return 0

def main():
    print(f"⏳ GPU {TARGET_GPU_ID}번에 {REQUIRED_FREE_VRAM}MB 이상의 여유 공간이 생기기를 기다리는 중...")
    
    while True:
        free_vram = get_free_vram(TARGET_GPU_ID)
        
        if free_vram >= REQUIRED_FREE_VRAM:
            print(f"\n🎉 목표 메모리 확보 완료! (현재 여유 메모리: {free_vram}MB)")
            print(f"🚀 명령어를 실행합니다: {' '.join(COMMAND)}\n")
            
            # 지정된 GPU만 보이도록 환경 변수 설정 후 코드 실행
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU_ID)
            
            # 코드 실행 및 끝날 때까지 대기
            subprocess.run(COMMAND, env=env)
            
            print("\n✅ 작업이 성공적으로 완료되었습니다!")
            break
        else:
            print(f"   [대기중] 현재 여유: {free_vram}MB / 필요: {REQUIRED_FREE_VRAM}MB ... ({CHECK_INTERVAL}초 후 재확인)", end='\r')
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()