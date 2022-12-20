"""
The text can be summarized by two ways: extractive text summarization
and abstractive text summarization.
the extractive test summarization identifies the important sentences and then add it to the summary 
and summary will contain text from original text.
the abstractive summarization understands the text well and intelligently generate the summary with new sentences

Transformers provides thousands of pretrained models to perform
tasks on different modalities such as text, vision, and audio.
These models can be applied on:
Text, for tasks like text classification, information 
extraction, question answering, summarization, 
translation, text generation, in over 100 languages.
Images, for tasks like image classification, object detection, 
and segmentation.
Audio, for tasks like speech recognition and audio classification.

PyTorch is an open source machine learning library used for developing
and training neural network based deep learning models.
PyTorch is designed to provide good flexibility 
and high speeds for deep neural network implementation.
It is open source, and is based on the popular Torch library.
PyTorch work on tensors,PyTorch dynamic computation graphs


Tensors are a type of data structure used in linear algebra, 
and like vectors and matrices,you can calculate arithmetic 
operations with tensors.
A tensor is a generalization of vectors and matrices 
and is easily understood as a multidimensional array.
A vector is a one-dimensional or first order tensor and 
a matrix is a two-dimensional or second order tensor.

"""
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# tokenizer allows us to convert sentences into set of tokens
# PegasusForConditionalGeneration allows us to use the model


# load tokenizer
tok = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
# using from_pretrained method to load up the existing model

# load the model
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
input_text = """
The Delhi University will today release the ‘special spot round’ of undergraduate admissions. Candidates who wish to get admitted to the varsity can check the special spot round list at the official DU website — admission.uod.ac.in.
Candidates will now have time from 10 am of December 19 till 11:59 pm on December 20 to submit applications. The special spot admission round allocated list will be announced on December 22 at 10 am by the varsity, and then the candidates can accept the allocated seat from 10 am of December 22 till 4:59 pm of December 23, 2022.
This list will be for the candidates who applied for admission to the varsity through CSAS 2022 but have not been yet admitted to any college. To be considered in the special spot admission round, the the admitted candidates will have to opt for ‘special spot admission’ through his/her dashboard on the official DU website.
Candidates should remember that it will be mandatory for candidates to take admission to the seat allocated in the special spot admission round. Any failure of acceptance of the allocated seat in the special spot admission round will forfeit the candidate’s eligibility for admission to the varsity. Also, there will be no option of upgrade and withdraw in this special spot admission round.

"""


# we need to convert our input_text to tokens. tokens are the number representations.

tokens = tok(input_text, truncation=True,
             padding="longest", return_tensors="pt")

# truncation is used to short our text and make sure it is of appropriate length
# and we mentioned to return pytroch tensors(pt)

# print(tokens)

# we need to summarize the input

encoded_text = model.generate(**tokens)

# simple way to pass the input ids and attention masks to the model
# by using  double asterisk (**) we are able to unpack the tokens and pass it to the model
# summarized-text consists of seperate set of tensors which is actually a output . we need to decode it to see it in humman readable form


# decode step

decoded = tok.decode(encoded_text[0], skip_special_tokens=True)
print(decoded)
