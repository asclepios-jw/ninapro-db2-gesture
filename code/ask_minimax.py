"""
외부검토요청_minimax.md 를 LLM API(OpenAI 호환)에 보내 검토 피드백을 받아 저장.

환경변수:
  NVIDIA_API_KEY (또는 MINIMAX_API_KEY / LLM_API_KEY)  ← 필수
  LLM_BASE_URL   기본 https://integrate.api.nvidia.com/v1   (엔드포인트에 맞게 변경)
  LLM_MODEL      기본 minimaxai/minimax-m1-40k             (실제 모델명으로 변경)

사용 (Jupyter):
  cell1:  %env NVIDIA_API_KEY=너의_키
  cell2:  %env LLM_MODEL=<모델명>          # 필요시
  cell3:  ! /usr/local/bin/python /mnt/.../code/ask_minimax.py
"""
import os, sys, json, time
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REVIEW_MD = os.path.join(PROJ, "외부검토요청_minimax.md")
OUT_DIR   = os.path.join(PROJ, "results", "minimax")
os.makedirs(OUT_DIR, exist_ok=True)

API_KEY  = (os.environ.get("NVIDIA_API_KEY") or os.environ.get("MINIMAX_API_KEY")
            or os.environ.get("LLM_API_KEY"))
BASE_URL = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
MODEL    = os.environ.get("LLM_MODEL", "minimaxai/minimax-m3")   # NVIDIA build 카탈로그 id


def die(msg):
    print(f"[에러] {msg}", file=sys.stderr); sys.exit(1)


def main():
    if not API_KEY:
        die("API 키가 없음. Jupyter에서  %env NVIDIA_API_KEY=너의_키  먼저 실행.\n"
            "       (! export 는 다음 셀로 안 넘어감 — %env 매직을 쓸 것.)")
    if not os.path.exists(REVIEW_MD):
        die(f"검토요청 파일 없음: {REVIEW_MD}")

    with open(REVIEW_MD, encoding="utf-8") as f:
        review = f.read()

    print(f"엔드포인트 : {BASE_URL}")
    print(f"모델       : {MODEL}")
    print(f"검토문서   : {len(review):,}자 전송\n요청 중...\n")

    MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "6000"))
    RETRIES    = int(os.environ.get("LLM_RETRIES", "4"))
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": "당신은 근전도(EMG) 기반 딥러닝 연구를 냉정하고 비판적으로 검토하는 "
                        "시니어 ML 리뷰어입니다. 칭찬이 아니라 결함·과대주장·놓친 접근을 지적하세요."},
            {"role": "user", "content": review},
        ],
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS,
        "stream": True,          # 추론모델은 스트리밍 필수 (non-stream은 read 타임아웃)
    }
    data = json.dumps(payload).encode("utf-8")

    def stream_once():
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=data,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
                     "Accept": "text/event-stream"},
            method="POST")
        cps, rps = [], []
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta", {})
                if delta.get("reasoning_content"):
                    rps.append(delta["reasoning_content"])
                if delta.get("content"):
                    cps.append(delta["content"]); print(delta["content"], end="", flush=True)
        return "".join(cps), "".join(rps)

    t0 = time.time()
    content = reasoning = ""
    for attempt in range(1, RETRIES + 1):
        print(f"(시도 {attempt}/{RETRIES} · max_tokens={MAX_TOKENS} — 첫 토큰까지 수십 초 대기 가능)\n")
        try:
            content, reasoning = stream_once()
            if content:
                break
            print("\n[빈 응답] 재시도...")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES:
                w = 8 * attempt
                print(f"\n[HTTP {e.code}] 서버 지연/과부하 — {w}s 후 재시도...")
                time.sleep(w); continue
            die(f"HTTP {e.code}: {body[:400]}  (401=키, 404=모델명, 5xx=서버지연)")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < RETRIES:
                w = 8 * attempt
                print(f"\n[타임아웃/네트워크: {e}] — {w}s 후 재시도...")
                time.sleep(w); continue
            die(f"네트워크/타임아웃: {e}")
    if not content:
        die("모든 재시도 실패. 잠시 후 다시 실행하거나 %env LLM_MAX_TOKENS=3000 으로 낮춰볼 것.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_txt = os.path.join(OUT_DIR, f"review_{ts}.md")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"# MiniMax 검토 응답 ({MODEL}, {ts})\n\n{content}\n")
        if reasoning:
            f.write(f"\n\n---\n## (모델 reasoning 과정)\n\n{reasoning}\n")
    print(f"\n\n{'='*70}\n[{time.time()-t0:.1f}s] 저장: {out_txt}")


if __name__ == "__main__":
    main()
