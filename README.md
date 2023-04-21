# Awesome Vision Language Prompt/Finetune/Adapter

## Introduction
With the rise of powerful pre-trained vision-language models like CLIP, the community has started to investigate potential solutions to efficiently adapt these models to downstream datasets and tasks. In order to systematically organize and understand the development of this field, we summarize awesome **Vision Language(VL) Prompt/Finetune/Adapter** methods and models. The list of papers is in chronological order.

## Timeline

[*2023*](#2023)

[*2022*](#2022)

[*2021*](#2021)

## 2023

1.  ****Dual Modality Prompt Tuning for Vision-Language Pre-Trained Model**** [*[CVPR]*](https://arxiv.org/abs/2208.08340) [*[code]*](https://github.com/fanrena/DPT)
    - Dual-modality Prompt Tuning (DPT) can learn text and visual prompts for both the text and image encoder simultaneously, equipped with a Class-Aware Visual Prompt Tuning(CAVPT) scheme to help the final obtained image feature concentrate more on the target visual concept.
    
    - Pre-Trained Model: CLIP
    
    - Datasets: EuroSAT, Caltech1-1, OxfordFlowers, Food101, FGVCAircraft, DTD, OxfordPets, StandfordCars, ImagNet1K, Sun397, UCF101
    
    - Domain Generalization(ImageNet, ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R) 
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled.png" /></p>
    
2.  ****Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning**** [*[NeurlPS]*](https://arxiv.org/abs/2210.08823) [*[code]*](https://github.com/dongzelian/SSF) 
    - SSF is a general proxy for parameter-efficient fine-tuning, where you only need to Scale and Shift your deep Features extracted by a pre-trained model for fine-tuning. It mainly has two merits: i)The scale and shift parameters do not depend on any input and have a unified learnable parameter space for different tasks. ii) It only introduces linear transformations.
    - Pretrained Models:
        - ViT-B/16
        - Swin-B
        - ConvNeXt-B
        - AS-MLP-B
    - Baselines:
        - 2 basic fine-tuning methods:
            - Full-Fine-Tuning
            - Linear-Probing
        - 2 Parameter-Efficient Fine-Tuning Methods:
            - Adapter
            - Bias
            - VPT
    - Tasks:
        - Performance Comparisions on Image Classification (FGVC, VTAB-1k, CIFAR-100, ImageNet-1K, totally 26 image classification tasks)
        - Impacts of Different Designs on SSF-ADA (CIFAR-100)
        - Performance Comparisons on Robustness and OOD Datasets (ImageNet-A, ImageNet-R, ImageNet-C, ImageNet-1K)
    <p align="center"><img width="30%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%201.png" /></p>
    
3.  ****Debiasing Vision-Language Models via Biased Prompts**** [*[arXiv]*](https://arxiv.org/abs/2302.00070) [*[code]*](https://github.com/chingyaoc/debias_vl)
    - Debias_VL is a general approach for self-debiasing foundation vision-language models by projecting out biased directions in the text embedding.
    
    - Pre-Trained Model: CLIP
    
    - Experiments
    
        - Discriminative models(zero-shot classifier, text-image retrieval)
        
        - Generative models(text-to-image)

    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%202.png" /></p>

4.  ****PLOT: Prompt Learning with Optimal Transport for Vision-Language Models**** [*[ICLR]*](https://arxiv.org/abs/2210.01253) [*[code]*](https://github.com/CHENGY12/PLOT)
    - PLOT is a prompt model based on CLIP and CoOp that uses optimal transport (OT) theory and two-stage optimization to learn multiple comprehensive prompts for describing different features of a category.
    - Pretrained Model: 
        - CLIP
        - CoOp
    - Task: 
        - few-shot recognition (Caltech101, ImageNet, OxfoldPets, StanfordCars, Flowers102, Food101, FGVCAircraft, DTD,  EuroSAT, UCF101, sun397)
        - domain generalization (ImageNet, ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R).
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%203.png" /></p>
    
5. ****VoLTA: Vision-Language Transformer with Weakly-Supervised Local-Feature Alignment**** [*[arXiv]*](https://arxiv.org/abs/2210.04135)
    - VoLTA (Vision-Language Transformer with weakly-supervised local-feature Alignment) is only utilizes image-caption data but achieves fine-grained region-level image understanding, eliminating the use of expensive box annotations.
    
    - Foundational Objective: Barlow Twins
    
    - Pre-Training & Downstream datasets: COCO, ImageNet, VOC07, LVIS, NLVR, Flicker30k
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%204.png" /></p>
    
6. ****CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment**** [*[ICLRr]*](https://arxiv.org/abs/2209.06430) [*[code]*](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP)
- CLIP-ViP model is proposed in three aspects:
    
    - adopt an image captioning model instead of using video captioning model
        
    - equip with Video Proxy mechanism
        
    - use Omnisource Cross-modal Learning(OCL)
- Preliminary:
    
    - post-pretraining with different data-scale
        
        - pre-trained models: CLIP-ViT-B/32, CLIP4Clip
            
        - dataset: WebVid-2.5M, HD-VILA-100M, HD-VILA-10M, MSR-VTT
            
    - language domain gap with downstream data
        
        - datasets: MSR-VTT, DiDeMo, HD-VILA-100M, webVid-2.5M, MS-COCO, Conceptual Caption 12M
            
        - pre-trained model: CLIP
 - Tasks:
    
    - Video-Text Post-Pretrainig(HD-VILA-100M)
        
    - Fine-tuning Training(MSR-VTT, DiDeMo, LSMDC, ActivityNet)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%205.png" /></p>
    
7. ****SgVA-CLIP: Semantic-guided Visual Adapting of Vision-Language Models for Few-shot Image Classification**** [*[arXiv]*](https://arxiv.org/abs/2211.16191)
    - Semantic-Guided Visual Adapting (SgVA) extends vision-language pre-trained models to produce discriminative adapted visual features with the guidance of the fine-grained cross-modal knowledge learned by the pre-trained model.
    
    - Baselines and Benchmarks
    
        - PEMnE-BMS\*, HCTransformers, CLIP_LP+LN, P>M>F, cluster-FSL, PT+MAP, EPNet and EASY(miniImagenet and tieredImagenet)
        
        - Zero-shot CLIP, CoOp, CLIP-Adapter, ProGrad(ImageNet, StandfordCars, UCF101, Caltech101, Flowers102, SUN397, DTD, EuroSAT, FGVCAircraft, OxfordPets, Food101)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%206.png" /></p>
    
8. ****Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models**** [*[arXiv]*](https://arxiv.org/abs/2211.02219) [*[code]*](https://tinyurl.com/mpe64f89)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%207.png" /></p>
    
9. ****Re-ViLM: Retrieval-Augmented Visual Language Model for Zero and Few-Shot Image Captioning**** [*[arXiv]*](https://arxiv.org/abs/2302.04858) 
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%208.png" /></p>
    
10. ****VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval**** [*[CVPR]*](https://arxiv.org/abs/2211.12764) [*[code]*](https://github.com/bighuang624/VoP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%209.png" /></p>
    
11. ****Contrastive Prompt Tuning Improves Generalization in Vision-Language Models**** [*[ICLR]*](https://openreview.net/forum?id=g4JB0ksCrKe)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2010.png" /></p>
    
12. ****Vision Transformer Adapter for Dense Predictions**** [*[ICLR]*](https://arxiv.org/abs/2205.08534) [*[code]*](https://github.com/czczup/ViT-Adapter)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2011.png" /></p>
    
13. ****T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models**** [*[arXiv]*](https://arxiv.org/abs/2302.08453) [*[code]*](https://github.com/TencentARC/T2I-Adapter)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2012.png" /></p>
    
14. ****Debiased Fine-Tuning for Vision-language Models by Prompt Regularization**** [*[arXiv]*](https://arxiv.org/abs/2301.12429)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2013.png" /></p>
    
15. ****Fine-tuned CLIP Models are Efficient Video Learners**** [*[CVPR]*](https://arxiv.org/abs/2212.03640) [*[code]*](https://github.com/muzairkhattak/ViFi-CLIP) 
    
    <p align="center"><img width="30%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2014.png" /></p>
16. ****Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models**** [*[CVPR]*](https://arxiv.org/abs/2301.06267) [*[code]*](https://github.com/linzhiqiu/cross_modal_adaptation)
    - The cross-modal adaptation approach treats examples from different modalities as additional few-shot examples, encoding different modalities to the same representation space.
    
    - Pre-trained Models:
    
        - CLIP
        
        - AudioCLIP
        
    - Task:
    
        - Vision-Language Adaption(Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101)
        
        - Vision-Audio Adaption(ImageNet, ESC-50)
    <p align="center"><img width=50% src="https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/assets/methodology.png" /></p>
17. ****Not All Features Matter: Enhancing Few-Shot CLIP with Adaptive Prior Refinement**** [*[arXiv]*](https://arxiv.org/pdf/2304.01195.pdf)[*[code]*](https://github.com/yangyangyang127/APE)
    - Adaptive Prior Refinement method (APE) directly utilizes the refined cache model for inference and explore the trilateral affinities  between the text image, the refined cache model and textual representations for robust training-free recognition.
    
    - Training-required APE-T simply trains lightweight category residuals on top other than costly fine-tuning the entire cache model.
    
    - Pre-Trained Models:
    
        - CLIP
        
        - CoOp
        
        - Tip-Adapter
        
    - Tasks
    
        - Comprehensive Evaluation(ImageNet, Caltech101, DTD, EuroSAT, FGVCAircraft, Flowers102, Food101, OxfordPets, StandfordCars, SUN397, UCF101)
        
        - Generalization Ability(ImageNet-V2, ImageNet-Sketch)
    <p align="center"><img width=50% src="https://github.com/yangyangyang127/APE/raw/main/framework.png" /></p>
18. ****Exploring Vision-Language Models for Imbalanced Learning**** [*[arXiv]*](https://arxiv.org/pdf/2304.01457.pdf) [*[code]*](https://github.com/Imbalance-VLM/Imbalance-VLM)
    - Imbalance-VLM uses supervised imbalanced methods in conjunction with VLMs to improve the performance of VLMs on tail classes, incorporating lightweight decoder after the ViT of VLMs to save memory and capture subtle features for tail classes.
    
    - Pre-Trained Models:
    
        - CLIP
        
        - Laion-CLIP
        
    - Datasets(ImageNet-LT, Places-LT, iNaturalist2018)
    <p align="center"><img width=50% src="https://github.com/Imbalance-VLM/Imbalance-VLM/raw/master/main-figure.png" /></p>
19. ****Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition**** [*[arXiv]*](https://arxiv.org/pdf/2304.04704.pdf) [*[code]*](https://github.com/amazon-science/prompt-pretraining)
    - POMP is a memory and computation efficient model and enables the learned prompt to condense semantic information for a rich set of visual concepts with over twenty-thousand classes.
    
    - Backbone: CLIP(ViT/B-16)
    
    - Dataset: ImageNet-21K
    <p align="center"><img width=50% src="https://github.com/amazon-science/prompt-pretraining/raw/main/docs/main_figure.png" /></p>
20. ****Chain of Thought Prompt Tuning for Vision-Language Models**** [*[arXiv]*] (https://arxiv.org/pdf/2304.07919.pdf)
    - Chain of Thought for prompt tuning combines visual and textual embeddings in vision domain and is consistent with the human learning paradigm, providing unique insight in vision domain.
    
    - Pre-Trained Model: CLIP
    
    - Datasets: ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN39, UCF101, DTD, EuroSAT
    
    - Tasks
    
        - Base-to-New Generalization
        
        - Cross-dataset Evaluation
        
        - Domain Gneralization
        
        - Image-Text Retrieval
        
        - Visual Question Answering
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitiled95.jpg" /></p>
21. ****Visual Instruction Tuning**** [*[arXiv]*](https://arxiv.org/pdf/2304.08485.pdf) [*[code]*](https://llava-vl.github.io/)
    - Large Language and Vision Assistant (LLaVA) is an end-to-end trained large multimodal model that connects the open-set visual  encoder of CLIP and large language models (LLM) for general purpose visual and language understanding.
    
    - Pre-Trained Model: CLIP
    
    - GPT-assisted Visual Instruction Data Generation: leverage language only GPT-4 or ChatGPT as the strong teacher to create instruction0following data involving visual content
    
        - Conversation
        
        - Detailed Description
        
        - Complex Reasoning
     <p align="center"><img width="50%" src="https://llava-vl.github.io/images/llava_arch.png" /></p>
22. ****Towards Robust Prompt on Vision-Language Models**** [*[arXiv]*](https://arxiv.org/pdf/2304.08479.pdf)
    - Robust Prompt Learning(ProL) improves robustness to both base and novel classes by integrating multi-scale features of an image into the prompt compared to existing in-context learning (IcoL) and ProL approaches, which is motivated by the robust multi-scale network architecture.
    
    - VLM: MEGMA(visual encoder NF_RN20x16 and language model GPT-Neo)
    
    - Datasets:
    
        - in-distribution data:ImageNet-1k
        
        - out-of-distribution(OOD) data: ImageNet-V2(re-collected ImageNet-like images), ImageNet-R(rendition images), ImageNet-C(natural corrupted images), ImageNet-S(sketch images), ImageNet-A(natural adversarial images)
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled96.jpg" /></p>
    
23. ****Progressive Visual Prompt Learning with Contrastive Feature Re-formation**** [*[arXiv]*](https://arxiv.org/pdf/2304.08386.pdf)
    - Progressive Visual Prompt (ProVP) demonstrates the effectiveness of visual prompts in V-L pre-trained models. It also prevents the serious deviation of the prompted visual feature form CLIP visual feature distribution.
    
    - Pre-Trained Model: CLIP
    
    - Tasks:
    
        - Few-Shot Learning(train on 1,2,4,8,shots and test on full test sets)
        
        - Base-to-Novel Generalization(train on 16 shots )
        
        - Datasets(11 benchmarks:ImageNet, Caltech101, FGVCAircraft, Flowers102, Food101, OxfordPets, StandfordCars, EuroSAT, DTD, SUN397, UCF101)
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled97.jpg" /></p>

## 2022

1. **Learning to Prompt for Continual Learning** [*[CVPR]*](https://arxiv.org/abs/2112.08654) [*[code]*](https://github.com/google-research/l2p)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2015.png" /></p>
    
2. **Visual Prompt Tuning** [*[ECCV]*](https://arxiv.org/abs/2203.12119) [*[code]*](https://github.com/kmnp/vpt)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2016.png" /></p>
    
3. **Unified Vision and Language Prompt Learning** [*[CVPR]*](https://arxiv.org/abs/2210.07225) [*[code]*](https://github.com/yuhangzang/UPT)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2017.png" /></p>
    
4. ****AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition**** [*[NeurlPS]*](https://arxiv.org/abs/2205.13535) [*[code]*](https://github.com/ShoufaChen/AdaptFormer)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2018.png" /></p>
    
5. ****Neural Prompt Search**** [*[arXiv]*](https://arxiv.org/abs/2206.04673) [*[code]*](https://github.com/ZhangYuanhan-AI/NOAH)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2019.png" /></p>
    
6. ****Convolutional Bypasses Are Better Vision Transformer Adapters**** [*[arXiv]*](https://arxiv.org/abs/2207.07039) [*[code]*](https://github.com/JieShibo/PETL-ViT) arXiv*
    <table><tr>
        <td>
            <img src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2020.png" border=0/></p>
        </td>
        <td>
            <img src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2021.png" border=0/></p>
        </td>
    </tr></table>
         
7. ****Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets**** [*[arXiv]*](https://arxiv.org/abs/2208.07463)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2022.png" /></p>
    
8. ****ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning**** [*[NeurlPS]*](https://arxiv.org/abs/2206.13559) [*[code]*](https://github.com/linziyi96/st-adapter)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2023.png" /></p>
    
9. ****Parameter-efficient Model Adaptation for Vision Transformers**** [*[arXiv]*](https://arxiv.org/abs/2203.16329)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2024.png" /></p>
    
10. ****VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks**** [*[CVPR]*](https://arxiv.org/abs/2112.06825) [*[code]*](https://github.com/ylsung/VL_adapter)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2025.png" /></p>
    
11. ****Prompt Vision Transformer for Domain Generalization**** [*[arXiv]*](https://arxiv.org/abs/2208.08914)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2026.png" /></p>     
12. ****Visual Prompt Tuning for Generative Transfer Learning**** [*[arXiv]*](https://arxiv.org/abs/2210.00990)
    
    <p align="center"><img width="30%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2027.png" /></p>
    
13. ****Learning Domain Invariant Prompt for Vision-Language Models**** [*[arXiv]*](https://arxiv.org/abs/2212.04196)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2028.png" /></p>
    
14. ****Domain-Unified Prompt Representations for Source-Free Domain Generalization**** [*[arXiv]*](https://arxiv.org/abs/2209.14926) [*[code]*](https://github.com/muse1998/Source-Free-Domain-Generalization)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2029.png" /></p>
    
15. ****Prompt-Matched Semantic Segmentation**** [*[arXiv]*](https://arxiv.org/abs/2208.10159)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2030.png" /></p>
    
16. ****Visual Prompting via Image Inpainting**** [*[arXiv]*](https://arxiv.org/abs/2209.00647) 
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2031.png" /></p>
    
17. ****Unleashing the Power of Visual Prompting At the Pixel Level**** [*[arXiv]*](https://arxiv.org/abs/2212.10556) [*[code]*](https://github.com/UCSC-VLAA/EVP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2032.png" /></p>
    
18. ****Exploring Visual Prompts for Adapting Large-Scale Models**** [*[arXiv]*](https://arxiv.org/abs/2203.17274) [*[code]*](http://hjbahng.github.io/visual_prompting)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2033.png" /></p>
    
19. ****Visual Prompt Tuning for Test-time Domain Adaptation**** [*[arXiv]*](https://arxiv.org/abs/2210.04831)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2034.png" /></p>
    
20. ****Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models**** [*[NeurlPS]*](https://arxiv.org/abs/2209.07511) [*[code]*](https://azshue.github.io/TPT)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2035.png" /></p>
    
21. ****Prompt Generation Networks for Efficient Adaptation of Frozen Vision Transformers**** [*[arXiv]*](https://arxiv.org/abs/2210.06466) [*[code]*](https://github.com/jochemloedeman/PGN)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2036.png" /></p>
    
22. **Multitask Vision-Language Prompt Tuning** [*[arXiv]*](https://arxiv.org/abs/2211.11720) [*[code]*](https://github.com/sIncerass/MVLPT)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2037.png" /></p>
    
23. ****Prompt Tuning with Soft Context Sharing for Vision-Language Models**** [*[arXiv]*](https://arxiv.org/abs/2208.13474)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2038.png" /></p>
    
24. ****Learning to Prompt for Vision-Language Models**** [*[IJCV]*](https://arxiv.org/abs/2109.01134) [*[code]*](https://github.com/KaiyangZhou/CoOp)
    - Based on continuous prompt learning and  provided 2 implementations that handle different tasks, Context Optimization(CoOp) models a prompt’s context words with learnable vectors while the entire pre-trained parameters are kept fixed, improving the deployment efficiency compared with proposed vision-language models.
    - Pretrained Models: CLIP
    - Tasks
        - Few-Shot Learning(ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101)
        - Domain Generalization(ImageNet, ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2039.png" /></p>
    
25. ****Language-Aware Soft Prompting for Vision & Language Foundation Models**** [*[arXiv]*](https://arxiv.org/abs/2210.01115)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2040.png" /></p>
    
26. ****Supporting Vision-Language Model Inference with Causality-pruning Knowledge Prompt**** [*[arXiv]*](https://arxiv.org/abs/2205.11100)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2041.png" /></p>
    
27. ****Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model**** [*[CVPR]*](https://arxiv.org/abs/2203.14940) [*[code]*](https://github.com/dyabel/detpro)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2042.png" /></p>
    
28. **A Good Prompt Is Worth Millions of Parameters: Low-resource Prompt-based Learning for Vision-Language Models** [*[ACL]*](https://arxiv.org/abs/2110.08484) [*[code]*](https://github.com/woojeongjin/FewVLM)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2043.png" /></p>
    
29. ****Prompting through Prototype: A Prototype-based Prompt Learning on Pretrained Vision-Language Models**** [*[arXiv]*](https://arxiv.org/abs/2210.10841)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2044.png" /></p>
    
30. ****Unsupervised Prompt Learning for Vision-Language Models**** [*[arXiv]*](https://arxiv.org/abs/2204.03649) [*[code]*](https://github.com/tonyhuang2022/UPL)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2045.png" /></p>
    
31. ****Prompt Distribution Learning**** [*[CVPR]*](https://arxiv.org/abs/2205.03340)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2046.png" /></p>
    
32. **Conditional Prompt Learning for Vision-Language Models** [*[CVPR]*](https://arxiv.org/abs/2203.05557) [*[code]*](https://github.com/KaiyangZhou/CoOp)
    - Conditional Context Optimization(CoCoOp) extends CoOp by further learning a lightweight neural network(Meta-Net) to generate for each image an input-conditional token(vector), allowing the gap between manual and learning-base prompts to be substantially reduced.
    - Pretrained Model: CLIP
    - Tasks:
        - Generalization from Base to New Classes(ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101)
        - Cross-Dataset Transfer(ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101)
        - Domain Generalization(ImageNet, ImageNetV2, ImageNet-Sketch, ImageNet-A, ImgaeNet-R)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2047.png" /></p>
    
33. ****DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting**** [*[CVPR]*](https://arxiv.org/abs/2112.01518) [*[code]*](https://github.com/raoyongming/DenseCLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2048.png" /></p>
    
34. ****CLIP also Understands Text: Prompting CLIP for Phrase Understanding**** [*[arXiv]*](https://arxiv.org/abs/2210.05836)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2049.png" /></p>
    
35. ****Bridge-Prompt: Towards Ordinal Action Understanding in Instructional Videos**** [*[CVPR]*](https://arxiv.org/abs/2203.14104) [*[code]*](https://github.com/ttlmh/Bridge-Prompt)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2050.png" /></p>
    
36. ****Prompting Visual-Language Models for Efficient Video Understanding**** [*[ECCV]*](https://arxiv.org/abs/2112.04478)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2051.png" /></p>
    
37. ****PointCLIP V2: Adapting CLIP for Powerful 3D Open-world Learning**** [*[CVPR]*](https://arxiv.org/abs/2211.11682) [*[code]*](https://github.com/yangyangyang127/PointCLIP_V2)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2052.png" /></p>
    
38. ****SVL-Adapter: Self-Supervised Adapter for Vision-Language Pretrained Models**** [*[BMV]*](https://arxiv.org/abs/2210.03794) [*[code]*](https://github.com/omipan/svl_adapter)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2053.png" /></p>
    
39. ****Localized Latent Updates for Fine-Tuning Vision-Language Models**** [*[arXiv]*](https://arxiv.org/abs/2212.06556)
40. ****EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning**** [*[arXiv]*](https://arxiv.org/abs/2210.07795)  [*[code]*](https://github.com/swaggy-TN/EfficientVLM)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2054.png" /></p>
    
41. ****Can Language Understand Depth?**** [*[ACM MM]*](https://arxiv.org/abs/2207.01077) [*[code]*](https://github.com/Adonis-galaxy/DepthCLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2055.png" /></p>
    
42. ****Prompting for Multi-Modal Tracking**** [*[ACM MM]*](https://arxiv.org/abs/2207.14571)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2056.png" /></p>
    
43. ****Expanding Language-Image Pretrained Models for General Video Recognition**** [*[ECCV]*](https://arxiv.org/abs/2208.02816) [*[code]*](https://aka.ms/X-CLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2057.png" /></p>
    
44. ****Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification**** [*[ECCV]*](https://arxiv.org/abs/2207.09519) [*[code]*](https://github.com/gaopengcuhk/Tip-Adapter) ECCV*
    - Tip-Adapter with Fine-tuning(Tip-Adapter-F) is the fine-tuned version of Tip-Adatper. It unfreezed the cached keys as a good initialization for learnable parameters and further fine-tuned them via SGD.
    
    - Pre-Trianed Models:
    
        - CLIP
        
        - CoOp
        
        - Tip-Adapter
        
    - Experiments(ImageNet, StandfordCars, UCF101,Caltech101, Flowers102, SUN397,DTD, EuroSAT, FGVCAircraft, OxfordPets, Food101)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2058.png" /></p>
    
45. ****Adapting CLIP For Phrase Localization Without Further Training**** [*[arXiv]*](https://arxiv.org/abs/2204.03647) [*[code]*](https://github.com/pals-ttic/adapting-CLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2059.png" /></p>
    
46. ****CPT: Colorful Prompt Tuning for Pre-trained Vision-Language Models**** [*[arXiv]*](https://arxiv.org/abs/2109.11797) [*[code]*](https://github.com/thunlp/CPT)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2060.png" /></p>
    
47. ****Domain Prompt Learning for Efficiently Adapting CLIP to Unseen Domains**** [*[arXiv]*](https://arxiv.org/abs/2111.12853) [*[code]*](https://github.com/shogi880/DPLCLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2061.png" /></p>
    
48. ****Clip-Tuning: Towards Derivative-free Prompt Learning with a Mixture of Rewards**** [*[EMNLP]*](https://arxiv.org/abs/2210.12050)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2062.png" /></p>
    
49. **Prompt-aligned Gradient for Prompt Tuning** [*[arXiv]*](https://arxiv.org/abs/2205.14865) [*[code]*](https://github.com/BeierZhu/Prompt-align)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2063.png" /></p>
    
50. ****DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations**** [*[arXiv]*](https://arxiv.org/abs/2206.09541)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2064.png" /></p>
    
51. ****Delving into the Openness of CLIP**** [*[arXiv]*](https://arxiv.org/abs/2206.01986)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2065.png" /></p>
    
52. ****OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression**** [*[NeurlPS]*](https://arxiv.org/abs/2206.02338)  [*[code]*](https://github.com/xk-huang/OrdinalCLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2066.png" /></p>
    
53. ****Prompt Tuning for Generative Multimodal Pretrained Models**** [*[arXiv]*](https://arxiv.org/abs/2208.02532) [*[code]*](https://github.com/OFA-Sys/OFA)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2067.png" /></p>
    
54. ****Contrastive Demonstration Tuning for Pre-trained Language Models**** [*[EMNLP]*](https://arxiv.org/abs/2204.04392) [*[code]*](https://github.com/zjunlp/PromptKG/tree/main/research/Demo-Tuning)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2068.png" /></p>
    
55. ****PPT: Pre-trained Prompt Tuning for Few-shot Learning**** [*[ACL]*](https://arxiv.org/abs/2109.04332) [*[code]*](http://github.com/thu-coai/PPT)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2069.png" /></p>
    
56. ****Pro-tuning: Unified Prompt Tuning for Vision Tasks**** [*[arXiv]*](https://arxiv.org/abs/2207.14381)

<p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2070.png" /></p>

56. ****MaPLe: Multi-modal Prompt Learning**** [*[arXiv]*](https://arxiv.org/abs/2210.03117) [*[code]*](https://tinyurl.com/2dzs8f3w)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2071.png" /></p>
    
57. ****Multi-Prompt Alignment for Multi-Source Unsupervised Domain Adaptation**** [*[arXiv]*](https://arxiv.org/abs/2209.15210)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2072.png" /></p>
    
58. ****An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA**** [*[AAAI]*](https://arxiv.org/abs/2109.05014) [*[code]*](https://github.com/microsoft/PICa)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2073.png" /></p>
    
59. ****VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning**** [*[CVPR]*](https://arxiv.org/abs/2102.10407) [*[code]*](https://github.com/Vision-CAIR/VisualGPT)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2074.png" /></p>
    
60. ****Flamingo: a Visual Language Model for Few-Shot Learning**** [*[arXiv]*](https://arxiv.org/abs/2204.14198)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2075.png" /></p>
    
61. ****Visual Clues: Bridging Vision and Language Foundations for Image Paragraph Captioning**** [*[arXiv]*](https://arxiv.org/abs/2206.01843)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2076.png" /></p>
    
62. ****DU-VLG: Unifying Vision-and-Language Generation via Dual Sequence-to-Sequence Pre-training**** [*[ACL]*](https://arxiv.org/abs/2203.09052)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2077.png" /></p>
    
63. ****Grounded Language-Image Pre-training**** [*[CVPR]*](https://arxiv.org/abs/2112.03857) [*[code]*](https://github.com/microsoft/GLIP) CVPR*
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2078.png" /></p>
    
64. ****GroupViT: Semantic Segmentation Emerges from Text Supervision**** [*[CVPR]*](https://arxiv.org/abs/2202.11094) [*[code]*](https://github.com/NVlabs/GroupViT)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2079.png" /></p>
    
65. ****Finetune like you pretrain: Improved finetuning of zero-shot vision models**** [*[arXiv]*](https://arxiv.org/abs/2212.00638) [*[code]*](https://github.com/locuslab/FLYP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2080.png" /></p>
    
66. ****CPL: Counterfactual Prompt Learning for Vision and Language Models**** [*[arXiv]*](https://arxiv.org/abs/2210.10362)*  [*[code]*](https://github.com/eric-ai-lab/CPL)  *arXiv*
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2081.png" /></p>
    
67. ****Zero-Shot Temporal Action Detection via Vision-Language Prompting**** [*[ECCV]*](https://arxiv.org/abs/2207.08184) [*[code]*](https://github.com/sauradip/STALE)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2082.png" /></p>
    

## 2021

1. ****AdaViT: Adaptive Vision Transformers for Efficient Image Recognition**** [*[CVPR]*](https://arxiv.org/abs/2111.15668)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2083.png" /></p>
    
2. ****Unified Multimodal Pre-training and Prompt-based Tuning for Vision-Language Understanding and Generation**** [*[arXiv]*](https://arxiv.org/abs/2112.05587)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2084.png" /></p>
    
3. **Learning Transferable Visual Models From Natural Language Supervision** [*[arXiv]*](https://arxiv.org/abs/2103.00020#) [*[code]*](https://github.com/OpenAI/CLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2085.png" /></p>
    
4. ****CLIP-Adapter: Better Vision-Language Models with Feature Adapters**** [*[arXiv]*](https://arxiv.org/abs/2110.04544) [*[code]*](https://github.com/OpenAI/CLIP)
    - CLIP-Adapter conducts residual-style feature blending to achieve efficient few-shot transfer learning via fine-tuning.
    
    - Baseline Models:
    
        - Linear probe CLIP
        
        - Zero-shot CLIP
        
        - CoOp
        
    - Experiments
    
        - Few-Shot Learning(ImageNet, StanfordCars, UCF101, Caltech101, Flowers102, SUN397, EuroSAT, FGVCAircraft, OxfordPets, Food101)
        
        - Visualization of Manifold(t-SNE, EuroSAT)
        
        - Ablation Studies(DTD, ImageNet)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2086.png" /></p>
    
5. **PointCLIP: Point Cloud Understanding by CLIP** [*[CVPR]*](https://arxiv.org/abs/2112.02413) [*[code]*](https://github.com/ZrrSkywalker/PointCLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2087.png" /></p>
    
6. ****Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling**** [*[arXiv]*](https://arxiv.org/abs/2111.03930) [*[code]*](https://github.com/gaopengcuhk/Tip-Adapter)
    - Trianing-Free CLIP-Adapter (Tip-Adapter) has strong performance on few-classification via directly setting the weights of adapter with a **cache model** to avoid the conventional SGD fine-tuning.
    
    - Pre-Trianed Models:
    
        - Zero-shot CLIP
        
        - Linear-porbe CLIP
        
        - CLIP-Adapter
        
        - CoOp
        
    - Tasks:
    
        - Efficiency Comparison(ImageNet, StandfordCars, UCF101, Caltech101, Flowers102, SUN397, DTD, EuroSAT, FGVCAircraft, OxfordPets, Food101)
        
        - Ablation Studies(ImageNet)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2088.png" /></p>
    
7. ****ActionCLIP: A New Paradigm for Video Action Recognition**** [*[arXiv]*](https://arxiv.org/abs/2109.08472) [*[code]*](https://github.com/sallymmx/ActionCLIP.git)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2089.png" /></p>
    
8. ****Multimodal Few-Shot Learning with Frozen Language Models**** [*[NeurlPS]*](https://arxiv.org/abs/2106.13884)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2090.png" /></p>
    
9. ****ClipCap: CLIP Prefix for Image Captioning**** [*[arXiv]*](https://arxiv.org/abs/2111.09734) [*[code]*](https://github.com/rmokady/CLIP_prefix_caption) arXiv*
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2091.png" /></p>
    
10. ****Unifying Vision-and-Language Tasks via Text Generation**** [*[ICML]*](https://arxiv.org/abs/2102.02779) [*[code]*](https://github.com/j-min/VL-T5)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2092.png" /></p>
    
11. **StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery** [*[ICCV]*](https://arxiv.org/abs/2103.17249) [*[code]*](https://github.com/orpatashnik/StyleCLIP)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2093.png" /></p>
    
12. ****Align and Prompt: Video-and-Language Pre-training with Entity Prompts**** [*[CVPR]*](https://arxiv.org/abs/2112.09583) [*[code]*](https://github.com/salesforce/ALPRO)
    
    <p align="center"><img width="50%" src="https://github.com/Hodasia/Awesome-Vision-Language-Finetune/blob/main/img/Untitled%2094.png" /></p>
