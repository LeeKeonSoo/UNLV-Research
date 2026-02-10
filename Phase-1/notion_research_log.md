# 연구 일지 - 2026년 2월 10일 (Week 4, Day 1)

## 📌 진행 상황 요약

**Current Status**: Subtask 1 - 기반 메트릭 파이프라인 구축 완료

**완료된 작업**:
- ✅ 프로젝트 구조 재정비 (legacy 폴더로 구현 이관)
- ✅ Khan Academy taxonomy 추출 파이프라인 구현
- ✅ Domain + Quality 메트릭 계산 파이프라인 구현
- ✅ Interactive dashboard 생성 스크립트 완성
- ✅ 전체 문서화 (README.md)

---

## 🎯 오늘의 핵심 결정사항

### 1. Graph 접근법 → Vector 기반 접근법으로 전환

**이전 계획**: Prerequisite graph를 명시적으로 구축

**새로운 접근**:
- Embedding similarity 기반 domain classification
- Multi-label soft assignment (multi-head attention 개념 차용)
- Cross-cutting concepts를 확률 분포로 표현

**이유**:
- Graph는 prerequisite 관계 모델링에 유용하지만, domain coverage 분석에는 과도하게 복잡
- Vector 기반 접근이 더 scalable하고 implementation이 간단
- Multi-domain 문서를 자연스럽게 처리 가능 (예: 한 문장이 math 60% + physics 40%)

### 2. 메트릭 우선순위: Domain + Quality 먼저

**선택한 메트릭**:
1. **Domain Coverage** (Multi-label classification)
   - Khan Academy taxonomy를 concept prototypes로 활용
   - Embedding similarity로 soft labels 할당
   - Top-5 concepts with normalized probabilities

2. **Quality Metrics**
   - Perplexity (GPT-2): 텍스트의 자연스러움 측정
   - Educational markers: examples, explanations, structure 검출

**미뤄진 메트릭** (Week 5-6에 추가):
- Difficulty (Flesch-Kincaid readability score)
- Redundancy (MinHash LSH for near-duplicate detection)

**이유**:
- 16주 중 4주차, 시간 제약
- Domain + Quality만으로도 충분한 insight 확보 가능
- 나머지는 점진적 추가 가능

### 3. 데이터셋 활용 전략: Khan + Tiny-Textbooks

**Khan Academy 역할**:
- Taxonomy source (structured labels)
- Concept prototype 생성 (embedding baselines)
- Ground truth for validation

**Tiny-Textbooks 역할**:
- Classification 대상 (unlabeled, high-quality)
- Khan taxonomy 적용 가능성 검증
- Real-world distribution 파악

**The Pile**: Week 7+ 이후 확장

---

## 🔬 구현된 파이프라인 아키텍처

### Step 1: Taxonomy Extraction (`1_extract_khan_taxonomy.py`)

**Input**: `khan_k12_concepts/all_k12_concepts.json` (982KB, 19 subjects)

**Process**:
```python
Khan Academy concepts
    ↓
Extract hierarchy (Subject → Grade → Concept)
    ↓
Embed each concept's article content
    ↓
Create concept prototypes (384-dim vectors)
```

**Output**:
- `outputs/khan_taxonomy.json` - Hierarchical structure
- `outputs/concept_prototypes.pkl` - Embeddings
- `outputs/metadata.json` - Statistics

**Embedding Model**: SentenceTransformer `all-MiniLM-L6-v2`
- Fast, lightweight (384 dimensions)
- Good balance of speed and quality
- Alternative models commented in code (Instructor, E5)

---

### Step 2: Metrics Computation (`2_compute_metrics.py`)

**Input**:
- Concept prototypes from Step 1
- Khan Academy full dataset
- Tiny-Textbooks (42 batches, ~420K docs)

**Process** (for each paragraph):
```python
Text chunk (200 words)
    ↓
Embed with SentenceTransformer
    ↓
Compute cosine similarity to all concept prototypes
    ↓
Top-5 domains with similarity > 0.3 → soft labels
    ↓
Compute perplexity with GPT-2
    ↓
Detect educational markers (regex patterns)
```

**Configuration**:
- `TOP_K_DOMAINS = 5` - Multi-label assignment
- `MIN_SIMILARITY = 0.3` - Threshold for relevance
- `CHUNK_SIZE = 200` - Words per paragraph

**Output**:
- `outputs/khan_analysis.jsonl` - Khan Academy results
- `outputs/tiny_textbooks_analysis.jsonl` - Tiny-Textbooks results

**Expected Runtime**:
- Khan Academy: ~10-15 minutes
- Tiny-Textbooks (full): ~1-2 hours on dual GPU (4060Ti + 3070Ti)

---

### Step 3: Dashboard Generation (`3_build_dashboard.py`)

**Input**: Analysis results from Step 2

**Aggregations**:
1. Domain distribution (subject-level counts)
2. Top 10 concepts by frequency
3. Quality statistics (mean/median perplexity, marker ratios)
4. Cross-cutting analysis (multi-domain percentage)

**Visualization**:
- Interactive HTML dashboard (Chart.js)
- Subject distribution comparison (bar chart)
- Educational markers comparison (bar chart)
- Top concepts (horizontal bar charts)
- Quality metrics table

**Output**: `outputs/dashboard.html` (self-contained, no server needed)

---

## 📊 예상 결과 (가설)

### Domain Coverage

**Khan Academy**:
- ✅ Well-balanced across K-12 subjects (Math, Science, Reading, History)
- ✅ Higher multi-domain ratio (cross-cutting concepts in FAQ format)
- ⚠️ Sparse in advanced topics (limited to K-12)

**Tiny-Textbooks**:
- ✅ More uniform distribution (GPT-generated diversity)
- ⚠️ Potential bias toward common/popular topics
- ❓ Lower multi-domain ratio? (textbook format = single topic focus)

### Quality Metrics

**Khan Academy**:
- ✅ Lower perplexity (~40-50) - human-written, curated
- ✅ High educational marker prevalence (examples, explanations)
- ✅ Consistent structure (FAQ format)

**Tiny-Textbooks**:
- ⚠️ Slightly higher perplexity (~50-60) - GPT artifacts
- ✅ High structure consistency (synthetic, templated)
- ❓ Fewer examples? (generated vs. human-crafted)

---

## 🚧 다음 단계 (Week 4-5)

### Immediate (This Week)
1. ✅ 코드 완성 및 문서화 (DONE)
2. ⏳ Step 1 실행: Taxonomy extraction (~10 minutes)
3. ⏳ Step 2 실행: Metrics computation (test with `max_batches=5` first)
4. ⏳ Step 3 실행: Dashboard generation
5. ⏳ 결과 검증: Manual inspection of 50-100 classified paragraphs

### Validation Strategy
- Sample 100 paragraphs from each dataset
- Manually label domain
- Compare with model predictions
- Compute precision/recall
- Use GPT-4 as second annotator for inter-annotator agreement

### Next Week (Week 5)
1. Full Tiny-Textbooks processing (all 42 batches)
2. Add difficulty metrics (Flesch-Kincaid)
3. Preliminary analysis write-up
4. Share dashboard with professor for feedback

---

## 💡 중요한 인사이트

### 1. SLM Training은 가능하다 (GPU 확인)
- 4060Ti (16GB) + 3070Ti (8GB) = 24GB VRAM
- 100M 모델 training은 가능 (LoRA/QLoRA 활용 시)
- 300M은 tight하지만 gradient checkpointing으로 가능

### 2. Dataset Characterization의 Novel Contribution
기존 연구 (Dolma, FineWeb, DataComp):
- ❌ High-level domain만 분류 (web/books/code)
- ❌ Fine-grained taxonomy 없음
- ❌ Cross-cutting concepts 미분석

우리 접근:
- ✅ Fine-grained taxonomy (K-12 curriculum 기반)
- ✅ Multi-label soft assignment
- ✅ Cross-cutting concept quantification
- ✅ Educational quality metrics

### 3. 시간 관리
- Week 4/16 = 25% 진행
- Subtask 1만 완료하기에도 빠듯
- Subtask 2 (model training)는 Week 11+ 이후 현실적
- Subtask 3 (refinement)는 "Future Work"로 처리 가능

---

## 🤔 여전히 미해결된 질문들

1. **Validation**: Domain classification accuracy는 얼마나 되는가?
   - Manual labeling으로 ground truth 생성 필요
   - 100개 sample로 precision/recall 측정

2. **Threshold 선택**: `MIN_SIMILARITY = 0.3`이 적절한가?
   - Too high → many unlabeled paragraphs
   - Too low → noisy labels
   - Validation 후 조정 필요

3. **The Pile 확장**: 어떻게 샘플링할 것인가?
   - Stratified by subset? (ArXiv, StackExchange, Books3, etc.)
   - Random 5GB sample?
   - 시간 제약 고려해야 함

4. **Prerequisite 관계**: 여전히 필요한가?
   - Curriculum ordering에는 필요
   - 하지만 co-occurrence 기반으로 간단히 처리 가능
   - Week 8-9에 재논의

---

## 📈 진행률 시각화

```
Week 1-3: [████████████████████] Data collection (완료)
Week 4:   [████████░░░░░░░░░░░░] Subtask 1 - Metrics pipeline (80% 완료)
Week 5:   [░░░░░░░░░░░░░░░░░░░░] Full analysis + validation (예정)
Week 6:   [░░░░░░░░░░░░░░░░░░░░] Difficulty + redundancy metrics (예정)
Week 7-8: [░░░░░░░░░░░░░░░░░░░░] The Pile sampling + analysis (예정)
Week 9-10:[░░░░░░░░░░░░░░░░░░░░] Prerequisite mining (예정)
Week 11+: [░░░░░░░░░░░░░░░░░░░░] Subtask 2 - Model training (미정)
```

---

## 🎓 예상 Contribution

**Conference Target**: COLM 2026 (August deadline) or EMNLP 2026

**Paper Angle**:
> "Fine-Grained Domain Characterization of Pretraining Corpora for Curriculum-Aware SLM Training"

**Key Claims**:
1. Existing datasets lack fine-grained domain analysis
2. Educational taxonomies can guide curriculum learning
3. Cross-cutting concepts are prevalent and measurable
4. Domain-balanced data improves SLM efficiency (if Subtask 2 완료 시)

**Without Subtask 2**: Workshop paper or short paper
**With Subtask 2**: Full conference paper

---

## 🔗 유용한 링크

- **Notion Page**: https://www.notion.so/Phase-1-2f5fa6116ae180a2bf73ccd81ad7ae8e
- **Khan Academy ToS**: https://www.khanacademy.org/about/tos
- **Tiny-Textbooks Dataset**: https://huggingface.co/datasets/nampdn-ai/tiny-textbooks
- **SentenceTransformers Docs**: https://www.sbert.net/

---

## ✅ Action Items for Tomorrow

1. [ ] Step 1 실행 (taxonomy extraction)
2. [ ] Step 2 테스트 실행 (`max_batches=5`)
3. [ ] Dashboard 생성 및 확인
4. [ ] Manual validation 샘플 100개 선정
5. [ ] Professor에게 진행상황 업데이트 이메일

---

**Last Updated**: 2026-02-10 19:30 KST
**Next Review**: 2026-02-11 (내일)
