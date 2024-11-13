# Awesome-Neuro-Symbolic-Learning-with-LLM

✨✨<b>Latest Advances on Neuro-Symbolic Learning in the Era of Foundation Models.</b>

<font size=5><b> Table of Contents </b></font>
- [Awesome-Neuro-Symbolic-Learning-with-LLM](#awesome-neuro-symbolic-learning-with-llm)
- [Awesome Tutorials \& Workshops](#awesome-tutorials--workshops)
- [Awesome Survey](#awesome-survey)
- [Awesome Papers](#awesome-papers)
  - [LLM for Neural-Symbolic Learning](#llm-for-neural-symbolic-learning)
  - [Nesy for LLM Reasoning \& Planning](#nesy-for-llm-reasoning--planning)
  - [Nesy for the Explainable \& Trustworthy](#nesy-for-the-explainable--trustworthy)
  - [Application](#application)
    - [Nesy for Visual Reasoning](#nesy-for-visual-reasoning)
    - [Nesy for Math](#nesy-for-math)
    - [Nesy for Agent](#nesy-for-agent)
    - [Nesy for RL](#nesy-for-rl)
    - [Nesy for Embodied AI](#nesy-for-embodied-ai)
    - [Nesy for AIGC](#nesy-for-aigc)
  - [Misc](#misc)
- [Awesome Datasets](#awesome-datasets)

# Awesome Tutorials & Workshops

- [Advances in Neuro Symbolic Reasoning and Learning](https://neurosymbolic.asu.edu/2023-aaai-tutorial-advances-in-neuro-symbolic-reasoning/) AAAI 2023
- [Neuro-Symbolic Approaches: Large Language Models + Tool Use](https://wenting-zhao.github.io/complex-reasoning-tutorial/slides/6.pdf) ACL 2023
- [Neuro-Symbolic Visual Reasoning and Program Synthesis](http://nscv.csail.mit.edu/) CVPR 2020
- [Neuro-Symbolic Learning and Reasoning in the Era of Large Language Models](https://nuclear-workshop.github.io/aaai2024/) Workshop in AAAI 2024
- [MIT 6.S191: Neuro-Symbolic AI](http://introtodeeplearning.com/2020/slides/6S191_MIT_DeepLearning_L7.pdf) Talk given by David Cox [Video](https://www.youtube.com/watch?v=4PuuziOgSU4)
- [Neuro-Symbolic Concepts for Robotic Manipulation](https://jiayuanm.com/data/2023-07-09-rss-neuro-symbolic-concepts.pdf) Talk given by Jiayuan Mao  [[Video]](https://www.youtube.com/watch?v=S8KsCtbJqz0)
- [Building General-Purpose Robots with Compositional Action Abstractions](https://jiayuanm.com/data/2024-04-19-brown-compositional-action-abstractions.pdf) Talk given by Jiayuan Mao  

# Awesome Survey
- [Neuro-Symbolic AI: The 3rd Wave](https://arxiv.org/pdf/2012.05876)
- [Neuro-Symbolic AI and its Taxonomy: A Survey](https://arxiv.org/pdf/2305.08876)
- [From Statistical Relational to Neuro-Symbolic Artificial Intelligence: a Survey](https://arxiv.org/pdf/2108.11451)
- [Neuro-Symbolic Artificial Intelligence: Current Trends](https://people.cs.ksu.edu/~hitzler/pub2/2021_AIC_NeSy.pdf)
  
# Awesome Papers
## LLM for Neural-Symbolic Learning
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**Evaluating Large Language Models Trained on Code**](https://arxiv.org/abs/2107.03374) <br> | Arxiv | 2021 | [Github](https://github.com/openai/human-eval) |
|[**MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning**](https://arxiv.org/pdf/2205.00445) <br> | Arxiv | 2022 | - |
|[**Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html) <br> | NeurIPS | 2023 |[Github](https://mvig-rhos.com/symbol_llm)|
|[**Large Language Models Are Neurosymbolic Reasoners**](https://arxiv.org/abs/2401.09334) <br> | AAAI | 2024 | [Github](https://github.com/hyintell/LLMSymbolic) |
|[**AutoSAT: Automatically Optimize SAT Solvers via Large Language Models**](https://arxiv.org/pdf/2402.10705) <br> | Arxiv | 2024 | - |


## Nesy for LLM Reasoning & Planning
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**PTR: Prompt Tuning with Rules for Text Classification**](https://arxiv.org/pdf/2105.11259) <br> | Arxiv | 2021 |[Github](https://github.com/thunlp/PTR)|
|[**Logic-Driven Context Extension and Data Augmentation for Logical Reasoning of Text**](https://aclanthology.org/2022.findings-acl.127/) <br> | ACL| 2022 | [Github](https://github.com/SiyuanWangw/LReasoner) |
|[**PAL: Program-aided Language Models**](https://arxiv.org/abs/2211.10435) <br> | ICML | 2023 | [Github](https://reasonwithpal.com/) |
|[**LLM Sandwich: NeuroSymbolic Approach to Solving Complex Reasoning Problem**](https://aclanthology.org/2023.emnlp-main.313.pdf) <br> | ACL | 2023 | [Github](https://github.com/benlipkin/linc) |
|[**Neuro-Symbolic Procedural Planning with Commonsense Prompting**](https://arxiv.org/pdf/2206.02928) <br> | ICLR | 2023 |[Github](https://github.com/YujieLu10/CLAP)|
|[**Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning**](https://arxiv.org/abs/2305.12295) <br> | EMNLP | 2023 |[Github](https://github.com/teacherpeterpan/Logic-LLM)|
|[**LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers**](https://aclanthology.org/2023.emnlp-main.313.pdf) <br> | EMNLP | 2023 |[Github](https://github.com/benlipkin/linc)|
|[**Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks**](https://arxiv.org/abs/2211.12588) <br> | TMLR | 2023 | [Github](https://github.com/TIGER-AI-Lab/Program-of-Thoughts) |
|[**Faithful Logical Reasoning via Symbolic Chain-of-Thought**](https://arxiv.org/abs/2405.18357) <br> | ACL | 2023 |[Github](https://github.com/Aiden0526/SymbCoT)|
|[**Binding Language Models in Symbolic Languages**](https://openreview.net/forum?id=lH1PV42cbF) <br> | ICLR | 2023 |[Github](https://lm-code-binder.github.io/)|
|[**SATLM: Satisfiability-Aided Language Models Using Declarative Prompting**](https://arxiv.org/pdf/2305.09656) <br> | NeurIPS | 2023 |[Github](https://github.com/xiye17/SAT-LM)|
|[**StackSight: Unveiling WebAssembly through Large Language Models and Neurosymbolic Chain-of-Thought Decompilation**](https://openreview.net/forum?id=gn5AsHIIwb) <br> | ICML | 2024 |-|
|[**Leveraging LLMs for Hypothetical Deduction in Logical Inference: A Neuro-Symbolic Approach**](https://arxiv.org/abs/2410.21779) <br> | Arxiv | 2024 | - |
|[**Natural Language Embedded Programs for Hybrid Language Symbolic Reasoning**](https://arxiv.org/abs/2309.10814) <br> | ACL | 2024 | [Github](https://github.com/luohongyin/LangCode) |
|[**Prototype-then-Refine: A Neurosymbolic Approach for Improved Logical Reasoning with LLMs**](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/BassemAkoushHashemElezabi.pdf) <br> | Arxiv | 2024 |-|
|[**Premise Order Matters in Reasoning with Large Language Models**](https://arxiv.org/abs/2402.08939) <br> | ICML | 2024 | - |
|[**LogicAsker: Evaluating and Improving the Logical Reasoning Ability of Large Language Models**](https://arxiv.org/abs/2401.00757) <br> | EMNLP | 2024 | [Github](https://github.com/yxwan123/LogicAsker) |
|[**Language Models can be Logical Solvers**](https://arxiv.org/pdf/2311.06158) <br> | ACL| 2024 | - |
|[**Abstract Meaning Representation-Based Logic-Driven Data Augmentation for Logical Reasoning**](https://arxiv.org/abs/2305.12599) <br> | ACL| 2024 | [Github](https://github.com/Strong-AI-Lab/Logical-Equivalence-driven-AMR-Data-Augmentation-for-Representation-Learning) |
|[**Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos**](https://openaccess.thecvf.com/content/CVPR2024/papers/Nagasinghe_Why_Not_Use_Your_Textbook_Knowledge-Enhanced_Procedure_Planning_of_Instructional_CVPR_2024_paper.pdf) <br> | CVPR | 2024 |[Github](https://github.com/Ravindu-Yasas-Nagasinghe/KEPP)|

## Nesy for the Explainable & Trustworthy
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**An Interpretable Neuro-Symbolic Reasoning Framework for Task-Oriented Dialogue Generation**](https://arxiv.org/abs/2203.05843) <br> | ACL| 2022 | [Github](https://github.com/shiquanyang/NS-Dial) |
|[**Bridging the Gap: Providing Post-Hoc Symbolic Explanations for Sequential Decision-Making Problems with Inscrutable Representations**](https://openreview.net/forum?id=o-1v9hdSult)<br> | NeurIPS | 2022 |-|
|[**Interpretable Neural-Symbolic Concept Reasoning**](https://proceedings.mlr.press/v202/barbiero23a/barbiero23a.pdf) <br> | ICML | 2023 |-|

## Application
### Nesy for Visual Reasoning 
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding**](https://arxiv.org/abs/1810.02338) <br> | NeurIPS | 2018 | [Github](https://github.com/kexinyi/ns-vqa) |
|[**Visual Concept-MetaConcept Learning**](https://arxiv.org/pdf/2002.01464) <br> | NeurIPS | 2019 |[Github](https://github.com/Glaciohound/VCML)|
|[**The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision**](http://nscl.csail.mit.edu/data/papers/2019ICLR-NSCL.pdf) <br> | ICLR | 2019 | [Github](https://github.com/vacancy/NSCL-PyTorch-Release) |
|[**Learning to Infer and Execute 3D Shape Programs**](https://arxiv.org/abs/1901.02875) <br> | ICLR | 2019 | [Github](https://github.com/HobbitLong/shape2prog) |
|[**FALCON: Fast Visual Concept Learning by Integrating Images, Linguistic descriptions, and Conceptual Relations**](https://arxiv.org/pdf/2203.16639)<br> | ICLR | 2022 | - |
|[**What's Left? Concept Grounding with Logic-Enhanced Foundation Models**](https://jiajunwu.com/papers/left_nips.pdf)<br> | NeurIPS | 2023 | [Github](https://github.com/joyhsu0504/LEFT) |
|[**NS3D: Neuro-Symbolic Grounding of 3D Objects and Relations**](https://arxiv.org/abs/2303.13483)<br> | CVPR | 2023 | [Github](https://github.com/joyhsu0504/NS3D) |
|[**GENOME: Generative Neuro-Symbolic Visual Reasoning by Growing and Reusing Modules**](https://openreview.net/forum?id=MNShbDSxKH) <br> | ICLR | 2024 |-|

### Nesy for Math 
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**A Symbolic Framework for Evaluating Mathematical Reasoning and Generalization with Transformers**](https://arxiv.org/abs/2305.12563) <br> | ACL | 2024 | [Github](https://github.com/jmeadows17/transformers-for-calculus) |
|[**AlphaIntegrator: Transformer Action Search for Symbolic Integration Proofs**](https://arxiv.org/abs/2410.02666)<br> | Arxiv| 2024 | - |
|[**Proving Olympiad Algebraic Inequalities without Human Demonstrations**](https://arxiv.org/abs/2406.14219)<br> | Arxiv| 2024 | - |
|[**Frugal LMs Trained to Invoke Symbolic Solvers Achieve Parameter-Efficient Arithmetic Reasoning**](https://arxiv.org/abs/2312.05571) <br> | AAAI | 2024 |-|

### Nesy for Agent
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**Symbolic Learning Enables Self-Evolving Agents**](https://arxiv.org/pdf/2406.18532v1)<br> | Arxiv| 2024 | [Github](https://github.com/aiwaves-cn/agents) |

### Nesy for RL
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**Neurosymbolic Reinforcement Learning with Formally Verified Exploration**](https://proceedings.neurips.cc/paper_files/paper/2020/file/448d5eda79895153938a8431919f4c9f-Paper.pdf)<br> | NeurIPS| 2020 | - |
|[**Neuro-Symbolic Reinforcement Learning with First-Order Logic**](https://aclanthology.org/2021.emnlp-main.283/) <br> | EMNLP | 2021 |-|
|[**Neurosymbolic Reinforcement Learning and Planning: A Survey**](https://arxiv.org/abs/2309.01038) <br> | IEEE TAI | 2023 |-|
|[**End-to-End Neuro-Symbolic Reinforcement Learning with Textual Explanations**](https://arxiv.org/abs/2403.12451) <br> | ICML | 2024 |[Github](https://ins-rl.github.io/)|

### Nesy for Embodied AI
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**PDSketch: Integrated Domain Programming, Learning, and Planning**](https://pdsketch.csail.mit.edu/data/papers/2022NeurIPS-PDSketch.pdf) <br> | NeurIPS| 2022 | [Github](https://github.com/vacancy/PDSketch-Alpha-Release) |
|[**JARVIS: A Neuro-Symbolic Commonsense Reasoning Framework for Conversational Embodied Agents**](https://arxiv.org/pdf/2208.13266)<br> | Arxiv| 2022 | - |
|[**Learning Neuro-Symbolic Skills for Bilevel Planning**](https://arxiv.org/abs/2206.10680)<br> | COIL | 2022 | [Github](https://github.com/Learning-and-Intelligent-Systems/predicators/releases/tag/skill-learning-june-2022) |
|[**ProgPrompt: Generating Situated Robot Task Plans using Large Language Models**](https://arxiv.org/abs/2209.11302) <br> | ICRA | 2023 |[Github](https://github.com/NVlabs/progprompt-vh)|
|[**What's Left? Concept Grounding with Logic-Enhanced Foundation Models**](https://jiajunwu.com/papers/left_nips.pdf)<br> | NeurIPS | 2023 | [Github](https://github.com/joyhsu0504/LEFT) |
|[**Programmatically Grounded, Compositionally Generalizable Robotic Manipulation**](https://arxiv.org/pdf/2304.13826) <br> | ICLR | 2023 |[Github](https://progport.github.io/)|
|[**Knowledge-based Embodied Question Answering**](https://arxiv.org/pdf/2109.07872) <br> | TPAMI | 2023 |-|
|[**Learning Adaptive Planning Representations with Natural Language Guidance**](https://arxiv.org/pdf/2312.08566) <br> | ICLR | 2024 |-|
|[**Learning Reusable Manipulation Strategies**](https://arxiv.org/pdf/2311.03293) <br> | CORL | 2023 |-|
|[**Compositional Diffusion-Based Continuous Constraint Solvers**](https://arxiv.org/abs/2309.00966) <br> | CORL | 2023 |[Github](https://github.com/zt-yang/diffusion-ccsp)|
|[**Hierarchical Planning and Learning for Robots in Stochastic Settings Using Zero-Shot Option Invention**](https://aair-lab.github.io/Publications/ss_aaai24.pdf) <br> | AAAI | 2024 |-|
|[**Learning Neuro-Symbolic Abstractions for Robot Planning and Learning**](https://ojs.aaai.org/index.php/AAAI/article/view/30409) <br> | AAAI | 2024 |-|
|[**Grounding Language Plans in Demonstrations through Counter-factual Perturbations**](https://arxiv.org/pdf/2403.17124) <br> | ICLR | 2024 |[Github](https://github.com/yanweiw/glide)|
|[**VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning**](https://arxiv.org/abs/2410.23156) <br> | Arxiv | 2024 |-|

### Nesy for AIGC 

## Misc
|  Title  |   Venue  |   Date   |   Code   |
|:--------|:--------:|:--------:|:--------:|
|[**On the Hardness of Probabilistic Neurosymbolic Learning**](https://proceedings.mlr.press/v235/maene24a.html) <br> | ICML | 2024 |-|
|[**On the Independence Assumption in Neurosymbolic Learning**](https://openreview.net/pdf?id=S1gSrruVd4) <br> | ICML | 2024 |-|
|[**Analysis for Abductive Learning and Neural-Symbolic Reasoning Shortcuts**](https://openreview.net/pdf?id=AQYabSOfci) <br> | ICML | 2024 |-|
|[**Convex and Bilevel Optimization for Neural-Symbolic Inference and Learning**](https://openreview.net/pdf?id=6NQ77Vj3DT) <br> | ICML | 2024 |-|
|[**Bridging Neural and Symbolic Representations with Transitional Dictionary Learning**](https://openreview.net/forum?id=uqxBTcWRnj) <br> | ICLR | 2024 |-|
|[**LogicMP: A Neuro-symbolic Approach for Encoding First-order Logic Constraints**](https://openreview.net/forum?id=BLGQ3oqldb) <br> | ICLR | 2024 |-|
|[**Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e560202b6e779a82478edb46c6f8f4dd-Abstract-Conference.html) <br> | NeurIPS | 2023 |-|
|[**Localized Symbolic Knowledge Distillation for Visual Commonsense Models**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/257be12f31dfa7cc158dda99822c6fd1-Abstract-Conference.html) <br> | NeurIPS | 2023 |-|
|[**A-NeSI: A Scalable Approximate Method for Probabilistic Neurosymbolic Inference**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4d9944ab3330fe6af8efb9260aa9f307-Abstract-Conference.html) <br> | NeurIPS | 2023 |-|
|[**Neuro-Symbolic Continual Learning:Knowledge, Reasoning Shortcuts and Concept Rehearsal**](https://proceedings.mlr.press/v202/marconato23a/marconato23a.pdf) <br> | ICML | 2023 | [Github](https://github.com/ema-marconato/NeSy-CL) |
|[**Out-of-Distribution Generalization by Neural-Symbolic Joint Training**](https://ojs.aaai.org/index.php/AAAI/article/view/26444) <br> | AAAI | 2023 |[Github](https://github.com/sbx126/NToC)|



# Awesome Datasets

- [CLEVR Dataset](https://cs.stanford.edu/people/jcjohns/clevr/) for VQA
- [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/) for VQA
- [LogiCity](https://jaraxxus-me.github.io/LogiCity/) for Abstract Urban Simulation
- [LogicGame](https://www.arxiv.org/pdf/2408.15778) Benchmarking Rule-Based Reasoning Abilities of Large Language Models
- [BabyAI](https://github.com/Farama-Foundation/Minigrid?tab=readme-ov-file)
- [Minecraft](https://github.com/tomsilver/pddlgym)
- [Mini-Behavior](https://github.com/StanfordVL/mini_behavior) for Embodied Tasks
- [CLIPort Dataset](https://cliport.github.io/) for Embodied Tasks
- [ALFworld](https://github.com/alfworld/alfworld) for Embodied Tasks
- [VirtualHome](http://virtual-home.org/) for Embodied Tasks
