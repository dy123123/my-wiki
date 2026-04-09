# llm-wiki

LLM이 관리하는 로컬 마크다운 위키. 문서를 업로드하면 LLM이 분석해서 위키를 자동으로 구성하고, 웹 UI와 MCP를 통해 AI 코딩 어시스턴트에서 바로 질문할 수 있습니다.

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/dy123123/my-wiki.git
cd my-wiki/llm-wiki
pip install ".[normalize-full,web]"
```

### 2. 환경 설정

```bash
cp .env.example .env
```

`.env` 파일에서 LLM 설정:

```bash
# LLM (MiniMax, OpenAI, Ollama 등 OpenAI 호환 API)
LLM_WIKI_LLM_BASE_URL=https://api.minimax.com/v1
LLM_WIKI_LLM_API_KEY=your-api-key
LLM_WIKI_LLM_MODEL=your-model-name

# 임베딩 (RAG용, 선택사항 — Qwen, nomic 등)
LLM_WIKI_EMBED_BASE_URL=http://localhost:11434/v1
LLM_WIKI_EMBED_MODEL=qwen3-embedding
LLM_WIKI_RERANK_MODEL=qwen3-reranker   # 선택사항
```

### 3. 초기화

```bash
llm-wiki init
```

### 4. 웹 서버 시작

```bash
llm-wiki web --port 7432 --token mysecret
```

브라우저에서 `http://서버IP:7432` 접속 → 토큰 입력 → 완료.

---

## 웹 UI에서 할 수 있는 것

| 탭 | 기능 |
|---|---|
| **Sources** | 파일 드래그&드롭 업로드 → 자동 분석 (normalize + ingest + embed) |
| **Wiki** | 생성된 위키 페이지 탐색 (디렉토리 트리 + 문서 간 링크) |
| **Ask** | 위키 + RAG 기반 질문 답변 |
| **Connect** | OpenCode / Claude Desktop / Cursor MCP 연결 설정 복사 |

파이프라인 상태는 소스마다 배지로 표시:
- **N** — Normalized (마크다운 변환 완료)
- **I** — Ingested (위키 페이지 생성 완료)
- **E** — Embedded (RAG 인덱스 완료)

---

## OpenCode / Claude Desktop MCP 연결

웹 UI의 **Connect** 탭에서 설정을 자동 생성해줌. 직접 설정하려면:

```json
{
  "mcpServers": {
    "my-wiki": {
      "type": "sse",
      "url": "http://서버IP:7432/mcp/sse",
      "headers": { "Authorization": "Bearer mysecret" }
    }
  }
}
```

연결하면 AI가 자동으로 `wiki_ask`, `wiki_search`, `wiki_page`, `wiki_status` 툴을 사용합니다.

---

## CLI 사용법 (선택사항)

웹 UI 대신 CLI로도 사용 가능:

```bash
# 문서 추가 + 한번에 처리
llm-wiki add paper.pdf
llm-wiki process --latest        # normalize + ingest + embed

# 질문
llm-wiki ask "UART TX 레지스터 주소는?"

# 검색
llm-wiki search "register address"

# 상태 확인
llm-wiki status
```

---

## 주요 환경변수

| 변수 | 설명 |
|---|---|
| `LLM_WIKI_LLM_BASE_URL` | LLM API 주소 (기본: OpenAI) |
| `LLM_WIKI_LLM_API_KEY` | API 키 |
| `LLM_WIKI_LLM_MODEL` | 모델명 |
| `LLM_WIKI_LLM_MAX_TOKENS` | 최대 토큰 수 (기본: 4096) |
| `LLM_WIKI_EMBED_MODEL` | 임베딩 모델 (비워두면 RAG 비활성화) |
| `LLM_WIKI_EMBED_BASE_URL` | 임베딩 API 주소 (비워두면 LLM 주소 사용) |
| `LLM_WIKI_RERANK_MODEL` | 리랭크 모델 (선택사항) |
| `LLM_WIKI_CHUNK_SIZE` | RAG 청크 크기 (기본: 800자) |
| `LLM_WIKI_VAULT_PATH` | 볼트 경로 (기본: `./vault`) |

`.env.example` 파일에 전체 옵션 있음.

---

## 볼트 구조

```
vault/
├── raw/          # 업로드된 원본 파일 (읽기 전용)
├── normalized/   # 마크다운으로 변환된 파일
├── chunks/       # RAG 청크 텍스트
├── embeddings/   # RAG 임베딩 벡터
└── wiki/
    ├── sources/  # 소스별 요약 페이지
    ├── entities/ # 인물, 조직, 제품
    ├── concepts/ # 기술 개념
    ├── topics/   # 주제
    └── analyses/ # 저장된 Q&A
```
