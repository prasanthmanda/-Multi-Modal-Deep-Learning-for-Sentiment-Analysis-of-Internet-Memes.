# Meme Sentiment Analysis
## CSE 676: Deep Learning Project

### Project Description
Memes are a popular method for spreading information and expressing sentiments on social media. This project aims to perform sentiment analysis on memes, considering their multimodal nature, which includes both images and text. We utilize deep learning models, specifically Bi-directional Long Short Term Memory (Bi-LSTM) and Gated Recurrent Unit (GRU) networks, to classify and quantify the humor and sentiment expressed in memes.

### Dataset
We use the [Memotion Dataset](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k), which contains approximately 7000 meme images with rich annotations. The dataset includes the following columns:
- `image_name`: Unique identifier for each image.
- `text_ocr`: Text extracted from the image using optical character recognition (OCR).
- `text_corrected`: Manually corrected text to better match the actual text on the meme.
- `Humor`: Categories like hilarious, not funny, very funny, etc.
- `Sarcasm`: Degrees of sarcasm such as very twisted, not sarcastic, twisted meaning, etc.
- `offensive`: Levels of offensiveness such as not offensive, extremely offensive, etc.
- `motivational`: Indicates whether the meme is motivational or not.
- `overall_sentiment`: Overall sentiment ranging from extremely positive to extremely negative.

### Background
Analyzing the sentiment of memes requires multimodal analysis, considering both text and images. Previous studies have shown that combining text and image modalities can enhance sentiment analysis performance. This project builds on such studies, using advanced deep learning techniques to analyze the sentiment expressed in memes.

### Preprocessing
#### Text Preprocessing
1. **Normalization**: Convert specific terms like URLs, emails, phone numbers, and dates into standard forms.
2. **Annotation**: Tag certain textual features in the output.
3. **Segmenter and Corrector**: Segment hashtags into meaningful words and correct spelling errors.
4. **Unpack Hashtags and Contractions**: Break down compound hashtags and expand contractions.
5. **Spell Correction**: Correct elongated words.
6. **Social Tokenizer**: Convert text to lowercase and tokenize it.
7. **Replacement Dictionaries**: Replace tokens with expressions for handling emoticons.

#### Chat-Based Text Preprocessing
1. **Chat Words Conversion**: Replace chat shortcuts with their expanded forms.
2. **Emoticons Conversion**: Replace emoticons with their descriptions.

#### Image Preprocessing
- Convert images to .png format.
- Extract features using the ResNet50 model and save them to disk to avoid recomputation.
- Load preprocessed and encoded image data for training and validation.

#### GloVe Vectors
- Use pre-trained GloVe word vectors for word embeddings, crucial for capturing semantic meanings in NLP tasks.

### Model Architecture
We use a neural network model combining Bi-LSTM for text data and a Dense layer for image features. The model merges the outputs from the text and image paths, applies fully connected layers, and uses activation functions like ReLU and softmax/sigmoid for classification.

### Training and Evaluation
We train separate models for each sentiment category: overall sentiment, humor, sarcasm, offensive, and motivational. The models are trained using the binary crossentropy loss function and the SGD optimizer. We evaluate the models based on validation accuracy.

### Real-World Application
Our sentiment analysis models for memes can be used in social media moderation and user engagement analytics, helping platforms understand and manage the emotional content of memes.


### References
- Dataset: https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k
- [Deep Learning for Sentiment Analysis: A Survey](https://arxiv.org/abs/1812.07883)
- [A Survey on Multimodal Sentiment Analysis](https://arxiv.org/abs/1908.11368)
- [SemEval-2020 Task 8: Memotion Analysis](https://arxiv.org/abs/2008.01745)
- [Understanding LSTM](https://arxiv.org/abs/1909.09586)
- [Sentiment Analysis using product review data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-015-0015-2)
- [Sentiment Analysis in Social Media: Systematic Literature Review](https://www.sciencedirect.com/science/article/pii/S1877050919317971)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations](https://arxiv.org/abs/1908.02265)
- [Memes in the Wild: Assessing the Generalizability of the Hateful Memes Challenge Dataset](https://arxiv.org/abs/2107.04313)
- [Guide on Word Embeddings in NLP](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
