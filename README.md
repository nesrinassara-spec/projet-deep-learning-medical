# Projet Deep Learning Médical — Classification des Fractures Osseuses

** Nesrine Gassara 2IDSD2
---

## C'est quoi ce projet ?

Dans ce projet j'ai essayé d'appliquer ce qu'on a vu en cours pour faire quelque chose d'utile dans le domaine médical.on donne une radiographie osseuse au modèle et il nous dit si c'est normal ou s'il y a une fracture.

Le dataset que j'ai utilisé s'appelle MURA-v1.1, c'est un dataset de Stanford qui contient plus de 40 000 radiographies de 7 parties du corps (épaule, coude, poignet, main, doigt, avant-bras, humérus).

---

## Ce que j'ai fait étape par étape

### 1. Exploration des données
Avant de commencer à coder les modèles j'ai d'abord regardé les données — combien d'images par classe, quelles parties du corps, est-ce qu'il y a un déséquilibre entre Normal et Fracture. J'ai fait des visualisations pour mieux comprendre le dataset.

### 2. Préparation des données
J'ai créé un Dataset personnalisé avec PyTorch (la classe qui hérite de torch.utils.data.Dataset). J'ai aussi ajouté de l'augmentation de données : rotation, flip, zoom aléatoire — ça permet au modèle de mieux généraliser et éviter l'overfitting.

### 3. Les modèles que j'ai construits

**Modèle 1 — CNN personnalisé**  
J'ai construit un réseau convolutionnel de zéro avec nn.Module. 4 blocs Conv → BatchNorm → ReLU → MaxPool puis un classifieur fully connected.

**Modèle 2 — ResNet18 avec Transfer Learning**  
J'ai utilisé un modèle déjà entraîné sur ImageNet et je l'ai adapté pour notre problème (1 canal gris, 2 classes). J'ai gelé les premières couches et réentraîné seulement la tête de classification.

**Modèle 3 — EfficientNet-B0**  
C'est une architecture plus récente et plus efficace. Ici j'ai fait un fine-tuning complet — toutes les couches sont entraînées mais avec un learning rate très faible pour ne pas perdre les features déjà apprises.

### 4. Entraînement
Pour l'entraînement j'ai appliqué la boucle standard PyTorch :
- zero_grad → forward → backward → step
- J'ai ajouté Early Stopping pour arrêter automatiquement si le modèle ne s'améliore plus
- Mixed Precision (AMP) pour accélérer sur GPU
- ReduceLROnPlateau pour réduire le learning rate automatiquement

### 5. Évaluation
J'ai évalué les 3 modèles avec plusieurs métriques : Accuracy, Precision, Recall, F1-Score, Spécificité et AUC-ROC. En médecine le Recall est la métrique la plus importante parce qu'on veut surtout éviter de rater une vraie fracture.

### 6. Grad-CAM
J'ai ajouté Grad-CAM qui est une technique d'interprétabilité — elle montre visuellement quelle zone de la radiographie le modèle regarde pour prendre sa décision. C'est très important en médecine pour vérifier que le modèle regarde bien l'os et pas autre chose.

### 7. Dashboard
J'ai créé un dashboard comparatif qui résume les performances des 3 modèles avec des graphiques et un tableau.

---

## Résultats

Les 3 modèles ont été comparés sur le test set. EfficientNet-B0 donne les meilleurs résultats grâce à son architecture optimisée, suivi de ResNet18. Le CNN personnalisé entraîné de zéro est moins bon ce qui confirme l'importance du Transfer Learning surtout quand les données médicales sont limitées.

---

## Technologies utilisées

- Python 3.10
- PyTorch 2.x
- torchvision
- scikit-learn
- Grad-CAM
- OpenCV
- Google Colab GPU T4
