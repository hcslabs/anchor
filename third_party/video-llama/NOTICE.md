# Video-LLaMA

Anchor's intake stage includes detecting and captioning candidate clips, and
filtering footage by prompt before it's worth running through reconstruction
— what the project page's lineage strip labels 人物・物体を分離
(person/object separation) and the broader Detect + Caption stages.

**Video-LLaMA** is a public, openly available instruction-tuned audio-visual
language model for video understanding that can perform this kind of
captioning, detection, and prompt-based filtering, and is referenced here as
representative of the class of open models this stage of the pipeline could
run on. Anchor's production intake currently runs on a newer pipeline of our
own — Video-LLaMA is not the model in production — but it's a fair, citable
reference point for what this stage of the architecture does and the kind of
open tooling that can do it.

**Source**

> [github.com/DAMO-NLP-SG/Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)

**Citation**

> Zhang, H., Li, X., Bing, L. *Video-LLaMA: An Instruction-tuned Audio-Visual
> Language Model for Video Understanding.* EMNLP 2023 (Demo).
> arXiv:[2306.02858](https://arxiv.org/abs/2306.02858)

No Video-LLaMA source code is redistributed in this repository. This notice
exists to credit a representative open model for the detection/captioning
class of tooling this pipeline stage uses. To clone the upstream repo locally
for reference:

```bash
git clone https://github.com/DAMO-NLP-SG/Video-LLaMA.git
```
