# 최종 결정사항 정리 (Final Decisions Summary)

**작성일**: 2026년 2월 10일
**연구자**: KeonSoo (bubbleguy10@gmail.com)

## Phase-1 Objective Sync (Canonical)
This decisions log is synchronized to `PHASE1_RESEARCH_OBJECTIVE_SPEC.md`.

1. Phase-1 claim scope is descriptive/comparative only.
2. Phase-1 excludes causal training-effect claims (Phase-2 responsibility).
3. Canonical RQ set:
   1. RQ1 domain coverage differences.
   2. RQ2 educational-structure quality differences.
   3. RQ3 metric reliability/stability.
4. Metric tiering is fixed:
   1. Core: domain, quality, difficulty.
   2. Exploratory: redundancy, perplexity.
5. Claiming policy:
   1. Only Core metrics can support headline conclusions.
   2. Exploratory metrics are diagnostic.
   3. Metrics failing reliability gates are non-claimable.

---

## 📋 프로젝트 핵심 결정사항

### 1. 연구 목표 및 범위

**최종 목표**: SLM(Small Language Model) 제작을 위한 데이터셋 특성 분석
- UNLV 연구실에서 진행 중 (Week 4/16)
- Phase-1: 데이터셋 characterization

**분석 대상 데이터셋**:
- ✅ Khan Academy K-12 (교육 콘텐츠 - 분류 기준 taxonomy 역할)
- ✅ Tiny-Textbooks (합성 교과서 데이터 - 분석 대상)

---

### 2. 기술적 접근 방식

#### 2.1 도메인 분류 방법론
**결정**: ~~Graph-based~~ → **Vector-based multi-label soft assignment**

**이유**:
- Cross-cutting concepts 처리 가능 (문서가 여러 도메인에 걸쳐 있는 경우)
- Multi-head attention과 유사한 확률적 접근
- 계산 효율성

**구현**:
- Khan Academy 개념을 concept prototypes로 사용
- TF-IDF 벡터화 후 cosine similarity로 top-5 도메인 할당
- Soft assignment (각 도메인별 확률 점수 제공)

---

#### 2.2 임베딩 방식
**결정**: ~~SentenceTransformers~~ → **TF-IDF (sklearn)**

**이유**:
- **원격 실행 환경 제약**: HuggingFace 모델 다운로드 불가 (403 Forbidden)
- 처음부터(밑바닥부터) 작동 가능해야 함
- 외부 의존성 최소화

**Trade-off**:
- 품질: TF-IDF < SentenceTransformers
- 실용성: TF-IDF 사용 가능 > SentenceTransformers 사용 불가

---

#### 2.3 품질 메트릭
**결정**: ~~Perplexity + Educational Markers~~ → **Educational Markers만**

**포함된 메트릭**:
1. **Domain Coverage** (도메인 커버리지)
   - Top-5 concept labels with scores

2. **Quality** (품질)
   - `has_examples`: "for example", "such as" 등
   - `has_explanation`: "because", "therefore" 등
   - `has_structure`: "first", "second", "in summary" 등

**제외된 메트릭**:
- ❌ Perplexity (GPT-2 다운로드 실패)
- ❌ Difficulty (향후 작업으로 보류)
- ❌ Redundancy (향후 작업으로 보류)

---

### 3. 실행 환경 요구사항

**핵심 제약**: "원격 컴퓨터로 실행하니까 밑바닥부터 시작해도 작동하게끔 해야 돼"

**의미**:
1. 로컬에 이미 수집된 데이터가 있는지 여부는 중요하지 않음
2. **코드를 실행해서 다운로드가 가능해야 함** (사용자 강조)
3. 외부 모델 다운로드 없이 작동해야 함 (HuggingFace, OpenAI API 등 불가)

**구현된 해결책**:
- TF-IDF 사용 (로컬 sklearn만 필요)
- Khan Academy API 대신 웹에서 다운로드 가능한 데이터셋 찾기
- Perplexity 계산 제거 (GPT-2 불필요)

---

### 4. 데이터 수집 전략

**사용자 지시**: "file을 막 찾지 말고 원하는 자료를 다운로드 받을 수 있는지 웹에서 찾아보는게 우선이지"

**변경사항**:
- ~~로컬 파일 시스템에서 기존 데이터 확인~~ (X)
- ~~Khan Academy API 직접 호출~~ (410 Gone - deprecated)
- **웹에서 다운로드 가능한 Khan Academy 데이터셋 검색** (진행 중)

**현재 상태**:
- HuggingFace `HuggingFaceTB/cosmopedia` 발견
- 403 Forbidden 오류로 다운로드 실패 (원격 프록시 제한)
- **대체 소스 필요** (현재 블로커)

---

### 5. 프로젝트 구조 및 파일 관리

**사용자 요청**: "불필요한 파일은 싹 다 legacy로 넣어버리고 필요한 코드와 충분한 설명의 md 문서만 남겨둬"

**최종 구조**:
```
Phase-1/
├── collect_khan_academy.py          # Step 0: 데이터 수집
├── collect_tinytextbooks.py
├── 1_extract_khan_taxonomy.py       # Step 1: TF-IDF 버전만
├── 2_compute_metrics.py             # Step 2: 간소화 버전만
├── 3_build_dashboard.py             # Step 3: 대시보드
├── config.py
├── utils.py
├── README.md                        # 전체 파이프라인 문서
├── notion_research_log.md           # 연구 일지
└── Claude.md                        # 현재 상황 문서
```

**Legacy로 이동**:
- `1_extract_khan_taxonomy_sbert.py` (SentenceTransformers 버전)
- `2_compute_metrics_sbert.py` (Perplexity 포함 버전)
- `SETUP.md` (중복 문서)
- `download_khan_data.py` (실패한 다운로드 시도)
- 이전 그래프 기반 실험 파일들

---

### 6. 문서화 요구사항

**사용자 지시**: "Claude.md 파일 만들어서 상황 작성하고 현재 상황을 나타내는 md문서 하나 만들고 맨 위에 변동사항이 생겼을때 수정하라고 문구로 지시해놔"

**생성된 문서**:
- ✅ `Claude.md`: 현재 프로젝트 전체 상황 (블로커, 파일 구조, 기술 결정, 다음 액션)
- ✅ 맨 위에 업데이트 지시사항 포함
- ✅ `notion_research_log.md`: 연구 일지 (이미 작성됨)
- ✅ `README.md`: 파이프라인 상세 문서

---

### 7. 분석 우선순위

**1차 목표** (현재 구현):
- Domain Coverage (도메인 분포)
- Quality (교육적 품질)

**2차 목표** (보류):
- Difficulty (난이도 분석)
- Redundancy (중복 제거)

**이유**:
- 일단 작동하는 파이프라인 완성 우선
- Khan Academy 데이터 문제 해결 후 확장

---

### 8. 파이프라인 설계 원칙

**결정된 원칙**:
1. **Remote-First**: 신규 환경에서도 작동
2. **Offline-Compatible**: 외부 모델 다운로드 불필요
3. **Incremental**: 각 단계가 재사용 가능한 artifact 생성
4. **Transparent**: 인터랙티브 대시보드로 결과 시각화

**단계별 출력물**:
- Step 0: `khan_k12_concepts/all_k12_concepts.json`
- Step 1: `outputs/khan_taxonomy.json`, `outputs/concept_prototypes_tfidf.pkl`
- Step 2: `outputs/khan_analysis.jsonl`, `outputs/tiny_textbooks_analysis.jsonl`
- Step 3: `outputs/dashboard.html`

---

## 🚨 현재 블로커 (Critical Issues)

### Khan Academy 데이터 다운로드 실패

**문제**:
- Khan Academy API: 410 Gone (deprecated)
- HuggingFace dataset: 403 Forbidden (원격 프록시 제한)

**사용자 강조**:
> "수집된게 중요한게 아니라 실행을 원격에서 하니까 수집부터 문제가 생기면 안된다고"

**영향**:
- 전체 파이프라인 블로킹
- Concept prototypes 생성 불가
- Domain classification 불가능

**다음 단계**:
- [ ] 웹에서 직접 다운로드 가능한 Khan Academy 데이터셋 찾기
- [ ] 대체 교육 콘텐츠 taxonomy 검토
- [ ] Mock data로 파이프라인 검증 후 실제 데이터 교체

---

## 📊 예상 결과물

### 완료 시 제공되는 것:
1. **Khan Academy taxonomy** (JSON)
2. **Concept prototypes** (TF-IDF vectors)
3. **Dataset analysis** (JSONL - 각 문단별 도메인 + 품질)
4. **Interactive dashboard** (HTML - Chart.js 시각화)

### 분석 인사이트:
- 어떤 도메인이 과다/과소 대표되는가?
- Tiny-Textbooks가 Khan Academy 대비 교육적 품질은?
- 몇 %의 콘텐츠가 설명 구조를 가지는가?

---

## 🎯 핵심 요약

**연구 질문**: "SLM pretraining을 위한 데이터셋을 어떻게 characterize할 것인가?"

**방법론**:
- Vector-based multi-label domain classification
- Educational markers detection
- TF-IDF (offline-compatible)

**현재 상태**:
- 파이프라인 코드 완성
- 프로젝트 정리 완료
- **블로킹**: Khan Academy 데이터 수집 실패

**긴급 액션**:
1. Khan Academy 데이터 다운로드 해결
2. End-to-end 파이프라인 검증
3. 결과 분석 및 다음 단계 계획

---

**마지막 업데이트**: 2026-02-10
**작성**: Claude (Sonnet 4.5) based on user's conversation history
