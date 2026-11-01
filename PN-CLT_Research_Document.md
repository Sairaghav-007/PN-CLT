# Prosody-Normalized Cross-Lingual Transfer (PN-CLT): A Lightweight Few-Shot Framework for Cross-Lingual Speech Emotion Recognition in Indian Languages

*(Structured to mirror the reference ESG/BRSR document format: Introduction → Clustered Literature Survey → Methodology Pipeline → Positioning Against Closest Work → References)*

---

## 1. Introduction

Speech Emotion Recognition (SER) automatically identifies a speaker's emotional state from vocal signals and underpins applications ranging from virtual assistants to mental-health monitoring and call-center analytics. Hybrid deep learning architectures — particularly CNN combined with Bidirectional LSTM — have substantially improved SER accuracy over traditional low-level acoustic feature methods, as demonstrated by Mishra et al. (2024), whose hybrid CNN-BiLSTM model achieved 96.36% accuracy on the English-language TESS and SAVEE corpora.

At the same time, the growth of high-accuracy SER models has not removed a fundamental generalization problem: models trained on English/Western speech data do not transfer reliably to other languages. The literature on cross-lingual SER shows that this failure is not simply a lack of labeled data; it also reflects the absence of a lightweight, transparent way to correct for prosodic and cultural variation between languages before or during transfer, as reflected in cross-corpus normalization work (Schuller et al., 2010; Feraru et al., 2015) and more recent cross-lingual transfer studies.

This issue is particularly significant for India, where over twenty officially recognized languages and hundreds of dialects each express emotion through markedly different pitch, intonation, rhythm and prosodic patterns. Documented cross-corpus experiments on Hindi, Urdu, Telugu and Kannada report accuracies of only 45–70% when models trained on other languages are applied without adaptation — far below the 90%+ accuracy typical of English-language SER.

Existing studies addressing this Indian-language gap have approached it through new feature types (Chakraborty et al., 2022), ensemble classifiers for a single language (Swain et al., 2022), data augmentation and feature fusion (Bangla ensemble work, 2025), or classical feature-selection pipelines (Tamil/Malayalam LightGBM work, 2025). However, these approaches usually address only one part of the problem. Some focus on a single Indian language without a transfer mechanism; some perform cross-lingual transfer without explicitly correcting for prosodic mismatch; and none combine a lightweight prosodic normalization step with few-shot deep transfer learning specifically for low-resource Indian regional languages.

The primary research gap adopted in this project is therefore methodological: the proposed contribution is to combine **language-adaptive prosodic normalization** with **few-shot cross-lingual transfer learning** on a hybrid CNN-BiLSTM backbone, within one tested framework (PN-CLT), rather than as two independent techniques. The project is applied to the CNN-BiLSTM architecture of Mishra et al. (2024) as its base, extended to Tamil, Telugu, Hindi and Kannada as target low-resource languages.

The current plan uses TESS/SAVEE as the English source corpus, a small few-shot Indian-language calibration set (50–100 samples per language), zero-shot baseline testing for problem quantification, and an ablation study to isolate the contribution of each PN-CLT component. This document describes the proposed method and its positioning against existing work; experimental validation is planned for the implementation phase.

---

## 2. Literature Survey (Clustered)

### A. Base hybrid SER architectures

| Author(s) & Year | Technology / Method | Year | Domain | What They Solved | Disadvantages / Limitations | Future Scope | Link |
|---|---|---|---|---|---|---|---|
| Mishra, Bhatnagar, Prakasam & Sureshkumar — Multimedia Tools and Applications (Springer) | Hybrid deep CNN + BiLSTM on MFSC features | 2024 | SER (English/Western) | Achieved 96.36% accuracy on TESS/SAVEE using CNN+BiLSTM feature fusion | Trained/validated only on English, Western-accent corpora; authors explicitly call SER an "open-ended problem" across regions/languages but do not test this themselves | Cross-lingual/cross-cultural validation (**this project's base paper and starting point**) | https://doi.org/10.1007/s11042-023-16849-x |

### B. Low-resource / Indian-language SER solutions

| Author(s) & Year | Technology / Method | Year | Domain | What They Solved | Disadvantages / Limitations | Future Scope | Link |
|---|---|---|---|---|---|---|---|
| Chakraborty, Dash, Panda & Solanki — ACM TALLIP | Phase-Based Cepstral Features (PBCC) + Gradient Boosting; cross-lingual transfer (Bengali↔English, SUBESCO-RAVDESS) | 2022 | Low-resource SER (Bengali) | Cross-lingual transfer achieving 86.9% (Bengali) / 82.7% (English) WA via a new feature type | Classical ML classifier (GBM), not deep learning; full-corpus transfer, not few-shot; no explicit prosodic normalization step; single Indian language only | Deep learning extension; few-shot settings; other Indian languages; explicit prosody correction (**directly extended by PN-CLT**) | https://doi.org/10.1145/3563944 |
| (Springer) — International Journal of Information Technology | ANN with STFT, chroma-VQT, Mel-spectrogram, MFCC ensemble on BanglaSER | 2024 | Single-language SER (Bengali) | Improved single-language accuracy via multi-feature ensemble | No cross-lingual component; no transfer learning; single language only | Cross-lingual extension | https://doi.org/10.1007/s41870-024-02310-1 |
| (PMC) — Bangla SER Ensemble Learning & Feature Fusion | Data augmentation + multi-stream deep learning feature fusion | 2025 | Single-language SER (Bengali) | Improved robustness via synthetic data augmentation and feature-level fusion | Speaker dependency and poor generalization remain unresolved even after augmentation; single language, no cross-lingual transfer | Cross-lingual generalization | PMC12387467 |
| Swain, Maji, Kabisatpathy et al. — Complex & Intelligent Systems (Springer) | DCRNN-based ensemble classifier | 2022 | Single-language SER (Odia) | First dedicated ensemble SER system for Odia | Single language; no transfer learning; no prosodic normalization | Extend to other Indian languages | https://doi.org/10.1007/s40747-022-00713-w |
| (ScienceDirect) — Reliable SER Framework for Multi-Regional Languages | Optimized LightGBM + feature selection | 2025 | Multi-regional SER (Tamil, Malayalam, English) | Improved accuracy via targeted feature selection for regional languages | Classical ML (LightGBM), not deep learning; no cross-lingual transfer mechanism; no prosody-specific correction | Deep learning extension; explicit cross-lingual transfer | ScienceDirect (2025) |
| (IIETA) — Indian Cross-Corpus SER Using Spectral-Temporal-Voice Quality Features | DCNN, multilingual training | 2024 | Diagnostic benchmark (Hindi, Urdu, Telugu, Kannada) | Quantified the cross-lingual accuracy gap: 58.83% (Hindi), 61.75% (Urdu), 69.75% (Telugu), 45.51% (Kannada) | Purely diagnostic — confirms the gap but proposes no correction method | A method to close this documented gap (**this is the empirical evidence PN-CLT is built to address**) | IIETA (2024) |
| Monisha, Tamanna & Sadia — Advances in Human-Computer Interaction (Hindawi) | Systematic review | 2022 | Review (Indo-Aryan & Dravidian language SER) | Comprehensively mapped the state of SER research for Indian language families, confirming the field is underexplored | Review only, no new model | Explicitly calls for more original SER work on Indian languages (**motivates this project**) | https://doi.org/10.1155/2022/9602429 |

### C. Cross-lingual transfer, domain adaptation and prosody normalization methods

| Author(s) & Year | Technology / Method | Year | Domain | What They Solved | Disadvantages / Limitations | Future Scope | Link |
|---|---|---|---|---|---|---|---|
| Schuller et al. | LLD feature normalization (prosody, voice quality) + linear SVM | 2010 | Cross-corpus SER (general) | Reduced feature mismatch across corpora via speaker-corpus normalization | Classical ML (SVM), pre-deep-learning; not language-specific; not combined with transfer learning | Extend normalization concept to modern deep hybrid architectures (**directly what PN-CLT does**) | — |
| Feraru et al. | LLD normalization (prosody, voice quality, spectral) + linear SVM | 2015 | Cross-corpus SER (general) | Similar normalization-based correction across labeled databases | Same limitations as Schuller et al.; no deep learning integration | Same as above | — |
| Zehra, Javed & Jalil — Complex & Intelligent Systems (Springer) | Ensemble learning across multiple corpora | 2021 | Cross-corpus multilingual SER | Improved generalization via ensembling multiple language corpora | Relies on ensemble averaging rather than explicitly correcting for prosodic variation | Explicit feature-level cross-lingual correction | https://doi.org/10.1007/s40747-020-00250-4 |
| Latif, Qadir & Bilal — ACII | Unsupervised adversarial domain adaptation | 2019 | Cross-lingual SER (Urdu vs. Western languages) | Domain adaptation without labeled target-language data, via adversarial training | Adversarial training adds architectural/training complexity; no explicit prosodic correction step; limited language pairs tested | Lighter-weight, more interpretable adaptation methods | — |
| Girish, Akhtar & Singh (NOVA-ARC) — arXiv preprint | Hyperbolic geometry + optimal-transport prototype alignment; non-verbal-to-verbal transfer | 2026 | Low-resource multilingual SER | Prosody-centric emotion transfer via geometry-aware representation learning | Architecturally complex (hyperbolic embeddings, optimal transport) — heavy for low-compute/practical deployment | Simpler, more computationally practical formulations (**PN-CLT's positioning relative to this closely-related 2026 work**) | arXiv:2604.17647 |
| (arXiv) — Learning Emotion-Discriminative Representations for Zero-Shot Cross-Lingual SER | Supervised contrastive learning + speaker-adversarial learning | 2026 | Zero-shot cross-lingual SER | Learned language-invariant, emotion-discriminative representations without target-language data | Zero-shot setting forgoes even minimal target supervision, trading some accuracy for generalization; complex dual-objective training | Few-shot variants combining light target-language supervision | arXiv:2606.06200 |
| (arXiv) — Pre-Finetuning for Few-Shot Emotional Speech Recognition | Wav2Vec2 multi-task pre-finetuning | 2023 | Few-shot SER (general) | Improved few-shot SER performance via SSL pre-finetuning across multiple datasets | Large SSL backbone (compute-heavy relative to CNN-BiLSTM); no prosodic normalization component; not Indian-language focused | Lighter-weight few-shot approaches for low-resource languages | arXiv:2302.12921 |

---

## 3. Methodology

The proposed methodology is a five-stage pipeline built around correcting and transferring the CNN-BiLSTM architecture of Mishra et al. (2024) for low-resource Indian languages. The technical status is draft: the stages are justified by the literature survey and problem statement, but exact normalization statistics, few-shot sample counts, and evaluation thresholds are to be finalized during implementation.

| Stage | Input | Method | Output | Purpose |
|---|---|---|---|---|
| **BASELINE** | English speech (TESS, SAVEE) | Reproduce Mishra et al.'s CNN-BiLSTM architecture | Baseline accuracy figures | Confirm reproducibility and establish a fair comparison point |
| **ZERO-SHOT TEST** | Indian-language speech (Tamil/Telugu/Hindi/Kannada) | Apply the unmodified baseline model directly, no adaptation | Accuracy drop measurement | Empirically quantify the cross-lingual generalization problem in your own experiments |
| **NORMALIZE** | Raw acoustic features (MFSC) per language | Language-adaptive prosodic normalization: rescale pitch, intonation, and rhythm features relative to each language's own baseline statistics, computed from a small calibration set | Normalized feature set per language | Correct for the documented prosodic/cultural mismatch identified as the root cause of cross-lingual failure |
| **TRANSFER** | Normalized features + small Indian-language labeled set (50–100 samples/language) | Few-shot fine-tuning: freeze the CNN-BiLSTM backbone (pretrained on English data), fine-tune only the final classification layers on the small target-language set | Adapted PN-CLT model per language | Address data scarcity directly, without requiring a large newly labeled Indian-language dataset |
| **EVALUATE & ABLATE** | PN-CLT vs. zero-shot baseline vs. fully-retrained baseline | Comparative accuracy/F1/data-efficiency evaluation; ablation with/without normalization layer and with/without few-shot head, isolated separately | Comparative results table + component-contribution breakdown | Prove overall improvement and identify which PN-CLT component drives it |

---

## 4. Positioning Against the Closest Existing Work

### 4.1 Against Chakraborty et al. (2022) — the closest prior Indian-language cross-lingual attempt

| Dimension | Chakraborty et al. (2022) | PN-CLT (this project) |
|---|---|---|
| Feature strategy | New feature type (PBCC) replacing MFCC | Standard MFSC features **plus** a normalization correction layer applied before the model — features aren't replaced, their distribution is corrected |
| Transfer mechanism | Full corpus-to-corpus transfer (SUBESCO↔RAVDESS) | Few-shot fine-tuning (50–100 samples/language) — targets languages with much less available labeled data than Bengali/SUBESCO |
| Classifier | Gradient Boosting (classical ML) | CNN-BiLSTM (deep learning) — captures temporal emotion dynamics GBM cannot |
| Target languages | Bengali only | Tamil, Telugu, Hindi, Kannada — none tested by Chakraborty et al. |
| Prosodic correction | Not explicit — relies on the new feature type itself to be more robust | Explicit, dedicated normalization step as a separate, measurable pipeline stage |

### 4.2 Against "Prosody as Supervision" / NOVA-ARC (2026) — the closest recent conceptual match

Both works target "prosody" + "low-resource multilingual SER," but the mechanisms differ substantially: NOVA-ARC uses hyperbolic-geometry embeddings and optimal-transport alignment for non-verbal-to-verbal supervision transfer — a sophisticated but computationally heavy approach. PN-CLT instead uses a simple, interpretable statistical normalization step combined with lightweight few-shot fine-tuning on a comparatively small CNN-BiLSTM backbone. This project should explicitly position itself as the **practical, lower-compute alternative** to NOVA-ARC-style approaches, rather than claiming to outperform it outright — the same "practicality over raw complexity" framing discussed in earlier analysis of this project.

### 4.3 Against Schuller (2010) / Feraru (2015) — prosody normalization precedent

These establish that prosody-based normalization for cross-corpus SER is not new in principle — PN-CLT's contribution is applying this established idea to a modern deep hybrid architecture, combined with few-shot transfer, for a population (Indian regional languages) neither paper addresses. This precedent must be cited explicitly in the related-work section to avoid any appearance of claiming normalization itself as original.

---

## 5. Feasibility & Plagiarism/Copyright Assessment

- **Feasibility:** High for a 3–6 month, single-author academic timeline. All components (Librosa/openSMILE feature extraction, CNN-BiLSTM via PyTorch, few-shot fine-tuning) are standard and run on free-tier compute (Google Colab/Kaggle).
- **Novelty safety:** No paper combines explicit prosodic normalization with few-shot CNN-BiLSTM transfer for Tamil/Telugu/Hindi/Kannada specifically. The closest works (Chakraborty et al. 2022; NOVA-ARC 2026) differ in mechanism, target languages, or both — documented in Section 4 above.
- **Plagiarism/copyright:** All datasets (TESS, SAVEE, AI4Bharat Indic speech corpora, Indian Emotional Speech Corpus) are public and appropriately licensed. Ensure the final manuscript paraphrases all descriptions of prior work in original language, and reproduces no code, text, or figures from cited papers directly.
- **Honest limitation to disclose in the paper itself:** normalization-based correction (Section 4.3) has a known ceiling relative to newer learned-representation approaches (Section 4.2); this project's contribution is best framed as a practical, resource-efficient method rather than a state-of-the-art accuracy claim.

---

## References

1. Mishra, S., Bhatnagar, N., Prakasam, P., Sureshkumar, T.R. (2024). Speech emotion recognition and classification using hybrid deep CNN and BiLSTM model. *Multimedia Tools and Applications*, 83, 37603–37620. https://doi.org/10.1007/s11042-023-16849-x
2. Chakraborty, C., Dash, T.K., Panda, G., Solanki, S.S. (2022). Phase-based cepstral features for automatic speech emotion recognition of low resource Indian languages. *ACM Transactions on Asian and Low-Resource Language Information Processing*. https://doi.org/10.1145/3563944
3. Enhanced artificial neural network-based SER model in low-resource Indian language. (2024). *International Journal of Information Technology* (Springer). https://doi.org/10.1007/s41870-024-02310-1
4. Bangla Speech Emotion Recognition Using Deep Learning-Based Ensemble Learning and Feature Fusion. (2025). PMC12387467.
5. Swain, M., Maji, B., Kabisatpathy, P. et al. (2022). A DCRNN-based ensemble classifier for speech emotion recognition in Odia language. *Complex & Intelligent Systems*, 8, 4237–4249. https://doi.org/10.1007/s40747-022-00713-w
6. A Reliable Speech Emotion Recognition Framework for Multi-Regional Languages Using Optimized LightGBM Classifier. (2025). ScienceDirect.
7. Indian Cross Corpus Speech Emotion Recognition Using Multiple Spectral-Temporal-Voice Quality Acoustic Features and DCNN. (2024). IIETA.
8. Monisha, A., Tamanna, S., Sadia, S. (2022). A review of the advancement in speech emotion recognition for Indo-Aryan and Dravidian Languages. *Advances in Human-Computer Interaction*, 2022, 9602429. https://doi.org/10.1155/2022/9602429
9. Schuller, B. et al. (2010). Cross-corpus acoustic emotion recognition: Variances and strategies (speaker-corpus normalization of prosodic/LLD features).
10. Feraru, S.M. et al. (2015). Cross-language acoustic emotion recognition: An overview and some tendencies (LLD normalization for cross-corpus SER).
11. Zehra, W., Javed, A.R., Jalil, Z. (2021). Cross corpus multi-lingual speech emotion recognition using ensemble learning. *Complex & Intelligent Systems*. https://doi.org/10.1007/s40747-020-00250-4
12. Latif, S., Qadir, J., Bilal, M. (2019). Unsupervised adversarial domain adaptation for cross-lingual speech emotion recognition. *8th International Conference on Affective Computing and Intelligent Interaction (ACII)*.
13. Girish, Akhtar, M.M., Singh, M. (2026). Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition. arXiv:2604.17647.
14. Learning Emotion-Discriminative Representations for Zero-Shot Cross-Lingual Speech Emotion Recognition. (2026). arXiv:2606.06200.
15. Pre-Finetuning for Few-Shot Emotional Speech Recognition. (2023). arXiv:2302.12921.

*Note: several entries above (Schuller 2010, Feraru 2015, IIETA 2024, ScienceDirect 2025) were identified via search snippets rather than direct publisher pages — verify full bibliographic details (exact title, volume, page numbers, DOI) against the publisher's own record before final submission.*
