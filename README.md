# Geoffrey Hinton Paper Replications

Educational **PyTorch** implementations of the most influential research papers by **Geoffrey Hinton and his collaborators**.

This repository follows the evolution of deep learning, from **Boltzmann Machines** and **Backpropagation** to **Deep Belief Networks**, **Dropout**, **Knowledge Distillation**, **Capsule Networks**, the **Forward-Forward Algorithm**, and **t-SNE**.

---

##  Goals

- Read and understand Geoffrey Hinton's foundational papers.
- Reproduce the core algorithms in PyTorch.
- Build clean, educational implementations.
- Visualize learning dynamics and learned representations.
- Document implementation details, assumptions, and simplifications.

---

# Paper Roadmap

# Architecture & Evolution Map

This diagram groups all 30 papers into the 8 core milestones highlighted in the repository overview. Click any node in GitHub to open its corresponding PyTorch notebook:

```mermaid
%%{init: {"theme": "default", "themeVariables": {"primaryColor": "#4a90e2", "secondaryColor": "#50e3c2", "lineColor": "#3399ff", "nodeBorder": "#3399ff"}}}%%
flowchart TD
    subgraph B1["1. Boltzmann Machines & Energy Foundations"]
        P01["01 (1977) Relaxation in Vision"] --> P02["02 (1985) Boltzmann Machines"]
        P02 --> P12["12 (1999) Products of Experts"]
        P12 --> P13["13 (2002) Contrastive Divergence"]
        P13 --> P16["16 (2007) RBMs for Collaborative Filtering"]
        P13 --> P19["19 (2010) ReLU Improves RBMs"]
    end

    subgraph B2["2. Backpropagation & Distributed Representations"]
        P03["03 (1986) Back-Propagating Errors"] --> P04["04 (1986) Distributed Representations"]
        P04 --> P05["05 (1986) Internal Reps by Error Prop"]
        P05 --> P07["07 (1992) Learning from Experience"]
    end

    subgraph B3["3. Deep Belief Nets & Generative Pre-Training"]
        P09["09 (1994) Autoencoders, MDL & Free Energy"] --> P10["10 (1995) The Helmholtz Machine"]
        P10 --> P11["11 (1995) The Wake-Sleep Algorithm"]
        P11 --> P14["14 (2006) Fast Learning for Deep Belief Nets"]
        P14 --> P15["15 (2006) Dimensionality Reduction (Deep AE)"]
        P14 --> P18["18 (2009) Deep Boltzmann Machines"]
        P14 --> P20["20 (2012) DNNs for Speech Recognition"]
    end

    subgraph B4["4. t-SNE & Manifold Visualization"]
        P08["08 (1992) Stochastic Neighbor Embedding"] --> P17["17 (2008) Visualizing Data using t-SNE"]
    end

    subgraph B5["5. Dropout, Scale & Deep Optimization"]
        P21["21 (2012) AlexNet Deep CNNs"] --> P22["22 (2013) Initialization and Momentum"]
        P22 --> P23["23 (2014) Dropout Regularization"]
        P23 --> P25["25 (2016) Layer Normalization"]
    end

    subgraph B6["6. Knowledge Distillation & Ensembles"]
        P06["06 (1991) Mixtures of Local Experts"] --> P24["24 (2015) Knowledge Distillation"]
    end

    subgraph B7["7. Capsule Networks & Part-Whole Hierarchies"]
        P27["27 (2017) Dynamic Routing Between Capsules"] --> P28["28 (2018) Matrix Capsules with EM Routing"]
        P28 --> P29["29 (2021) Part-Whole Hierarchies (GLOM)"]
    end

    subgraph B8["8. Forward-Forward & Biologically Plausible Learning"]
        P26["26 (2016) Fast Weights Attention"] --> P30["30 (2022) The Forward-Forward Algorithm"]
    end

    %% Cross-Era Evolutionary Links
    P13 ==> P14
    P07 ==> P15
    P15 ==> P17
    P20 ==> P21
    P23 ==> P24
    P21 ==> P27
    P25 ==> P26
    P02 -.-> P30

    %% Clickable Repository Links
    click P01 "./Relaxation_and_its_Role_in_Vision.ipynb"
    click P02 "./A_Learning_Algorithm_for_Boltzmann_Machines.ipynb"
    click P03 "./Learning_representations_by_back_propagating_errors.ipynb"
    click P04 "./Distributed_Representations.ipynb"
    click P05 "./Learning_Internal_Representations_by_Error_Propagation.ipynb"
    click P06 "./Adaptive_Mixtures_of_Local_Experts.ipynb"
    click P07 "./How_Neural_Networks_Learn_from_Experience.ipynb"
    click P08 "./Stochastic_Neighbor_Embedding.ipynb"
    click P09 "./Autoencoders_Minimum_Description_Length_and_Helmholtz_Free_Energy.ipynb"
    click P10 "./The_Helmholtz_Machine.ipynb"
    click P11 "./The_wake_sleep_algorithm_for_unsupervised_neural_networks.ipynb"
    click P12 "./Products_of_Experts.ipynb"
    click P13 "./Training_Products_of_Experts_by_Minimizing_Contrastive_Divergence.ipynb"
    click P14 "./A_fast_learning_algorithm_for_deep_belief_nets.ipynb"
    click P15 "./Reducing_the_Dimensionality_of_Data_with_Neural_Networks.ipynb"
    click P16 "./Restricted_Boltzmann_Machines_for_Collaborative_Filtering.ipynb"
    click P17 "./Visualizing_Data_using_t_SNE.ipynb"
    click P18 "./Deep_Boltzmann_Machines.ipynb"
    click P19 "./Rectified_Linear_Units_Improve_Restricted_Boltzmann_Machines.ipynb"
    click P20 "./Deep_Neural_Networks_for_Acoustic_Modeling_in_Speech_Recognition.ipynb"
    click P21 "./ImageNet_Classification_with_Deep_Convolutional_Neural_Networks_AlexNet.ipynb"
    click P22 "./On_the_importance_of_initialization_and_momentum_in_deep_learning.ipynb"
    click P23 "./Dropout_A_Simple_Way_to_Prevent_Neural_Networks_from_overfitting.ipynb"
    click P24 "./Distilling_the_Knowledge_in_a_Neural_Network.ipynb"
    click P25 "./Layer_Normalization.ipynb"
    click P26 "./Using_Fast_Weights_to_Attend_to_the_Recent_Past.ipynb"
    click P27 "./Dynamic_Routing_Between_Capsules.ipynb"
    click P28 "./Matrix_Capsules_with_EM_Routing.ipynb"
    click P29 "./How_to_Represent_Part_Whole_Hierarchies_in_a_Neural_Network.ipynb"
    click P30 "./The_Forward_Forward_Algorithm_Some_Preliminary_Investigations.ipynb"
%% Highlight the 11 Recommended Learning Path Milestones
    classDef learningPath stroke:#f5a623,stroke-width:3px,color:#000,fill:#fff8e7
    class P03,P02,P13,P14,P15,P17,P21,P23,P24,P27,P30 learningPath
```

---

# Recommended Learning Path

If you're new to Geoffrey Hinton's work, a good progression is:

If you're new to Geoffrey Hinton's work, follow this 11-step progression through the core breakthroughs (highlighted in gold in the

[Architecture & Evolution Map](#architecture--evolution-map) above):

| Step | Milestone | Diagram Bucket & Node | Primary PyTorch Notebook | Paper Archive | Core Algorithmic Focus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Backpropagation** | [Bucket 2 · `P03`](#architecture--evolution-map) | [📘 `#03` Back-Propagating Errors](./Learning_representations_by_back_propagating_errors.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Multi-layer chain rule, hidden representation learning, and gradient flow. |
| **2** | **Boltzmann Machines** | [Bucket 1 · `P02`](#architecture--evolution-map) | [📘 `#02` Boltzmann Machines](./A_Learning_Algorithm_for_Boltzmann_Machines.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Energy-based models, stochastic binary units, and thermal equilibrium. |
| **3** | **Contrastive Divergence** | [Bucket 1 · `P13`](#architecture--evolution-map) | [📘 `#13` Contrastive Divergence](./Training_Products_of_Experts_by_Minimizing_Contrastive_Divergence.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Tractable $k$-step MCMC approximation ($CD\text{-}k$) for training RBMs. |
| **4** | **Deep Belief Networks** | [Bucket 3 · `P14`](#architecture--evolution-map) | [📘 `#14` Deep Belief Nets](./A_fast_learning_algorithm_for_deep_belief_nets.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Greedy layer-wise unsupervised RBM pre-training followed by fine-tuning. |
| **5** | **Deep Autoencoders** | [Bucket 3 · `P15`](#architecture--evolution-map) | [📘 `#15` Dimensionality Reduction](./Reducing_the_Dimensionality_of_Data_with_Neural_Networks.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Non-linear manifold compression via RBM-initialized encoder-decoder bottlenecks. |
| **6** | **t-SNE** | [Bucket 4 · `P17`](#architecture--evolution-map) | [📘 `#17` Visualizing Data using t-SNE](./Visualizing_Data_using_t_SNE.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Student-$t$ kernel neighborhood probabilities to resolve the crowding problem. |
| **7** | **AlexNet** | [Bucket 5 · `P21`](#architecture--evolution-map) | [📘 `#21` AlexNet Deep CNNs](./ImageNet_Classification_with_Deep_Convolutional_Neural_Networks_AlexNet.ipynb) | [📄 PDF / Manifest](./resources/papers/) | End-to-end supervised ReLU CNNs, local response normalization, and data augmentation. |
| **8** | **Dropout** | [Bucket 5 · `P23`](#architecture--evolution-map) | [📘 `#23` Dropout Regularization](./Dropout_A_Simple_Way_to_Prevent_Neural_Networks_from_overfitting.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Stochastic sub-network sampling to prevent hidden feature co-adaptation. |
| **9** | **Knowledge Distillation** | [Bucket 6 · `P24`](#architecture--evolution-map) | [📘 `#24` Knowledge Distillation](./Distilling_the_Knowledge_in_a_Neural_Network.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Temperature-scaled softmax ("dark knowledge") transfer from teacher to student. |
| **10** | **Capsule Networks** | [Bucket 7 · `P27`](#architecture--evolution-map) | [📘 `#27` Dynamic Routing Capsules](./Dynamic_Routing_Between_Capsules.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Vector-output capsules, squashing non-linearity, and iterative routing-by-agreement. |
| **11** | **Forward-Forward Algorithm** | [Bucket 8 · `P30`](#architecture--evolution-map) | [📘 `#30` Forward-Forward Algorithm](./The_Forward_Forward_Algorithm_Some_Preliminary_Investigations.ipynb) | [📄 PDF / Manifest](./resources/papers/) | Layer-local positive vs. negative goodness passes without backward gradient propagation. |


# Philosophy

The goal of this repository is **education rather than benchmark reproduction**.

Implementations prioritize:

- readability
- modular code
- mathematical clarity
- reproducibility
- visualization
- extensive comments

Whenever the original paper uses computationally expensive experiments, simplified versions are provided while preserving the core algorithmic ideas.

---

# Disclaimer

These implementations may use:

- smaller datasets
- reduced model sizes
- fewer training epochs
- simplified hyperparameters

They should be considered **educational reproductions**, not exact replications of the original experimental results.

---

# References

Each project includes:

- Original paper
- Mathematical derivations
- Implementation notes
- Additional resources when appropriate

---

# Author

**Mohammed Fahd Abrah**

AI Engineer • Technical Writer

---

# License

This project is released under the **MIT License**.
