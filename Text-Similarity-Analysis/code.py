import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import time

gpt2_model_name = "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"
bert_model_name = "ytu-ce-cosmos/turkish-base-bert-uncased"
tokenizer_gpt2 = AutoTokenizer.from_pretrained(gpt2_model_name)
model_gpt2 = AutoModel.from_pretrained(gpt2_model_name)
tokenizer_bert = AutoTokenizer.from_pretrained(bert_model_name)
model_bert = AutoModel.from_pretrained(bert_model_name)

# @brief Extracts the embedding of a given text using a tokenizer and model.
# @param text -> the input text to be embedded
# @param tokenizer -> the tokenizer used for processing the text
# @param model -> the model used to generate embeddings
# @param embedding_cache -> a dictionary to cache embeddings for faster retrieval
# @return the embedding vector for the input text
def get_embedding(text, tokenizer, model, embedding_cache):
    if text in embedding_cache:
        return embedding_cache[text]

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    embedding = hidden_states.mean(dim=1).detach().numpy().flatten()
    # print(text)

    embedding_cache[text] = embedding
    return embedding

sample_data = pd.read_excel("soru_cevap.xlsx")
data = sample_data.sample(n=1000, random_state=42)

gpt2_embedding_cache = {}
bert_embedding_cache = {}
human_responses = data["insan cevabı"].tolist()
machine_responses = data["makine cevabı"].tolist()
questions = data["soru"].tolist()

bert_human_response_embeddings = [get_embedding(response, tokenizer_bert, model_bert, bert_embedding_cache) for response in human_responses]
bert_machine_response_embeddings = [get_embedding(response, tokenizer_bert, model_bert, bert_embedding_cache) for response in machine_responses]
bert_question_embeddings = [get_embedding(question, tokenizer_bert, model_bert, bert_embedding_cache) for question in questions]
# gpt2_human_response_embeddings = [get_embedding(response, tokenizer_gpt2, model_gpt2, gpt2_embedding_cache) for response in human_responses]
# gpt2_machine_response_embeddings = [get_embedding(response, tokenizer_gpt2, model_gpt2, gpt2_embedding_cache) for response in machine_responses]
# gpt2_question_embeddings = [get_embedding(question, tokenizer_gpt2, model_gpt2, gpt2_embedding_cache) for question in questions]

# @brief Calculates the cosine similarity between two embedding vectors.
# @param embedding1 -> the first embedding vector
# @param embedding2 -> the second embedding vector
# @return the cosine similarity score between the two embeddings
def get_cosine_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    # print(similarity)
    return similarity


# @brief Finds the most similar responses to a given question based on cosine similarity.
# @param question_embedding -> the embedding of the question
# @param responses -> a list of response texts
# @param response_embeddings -> a list of embeddings corresponding to the responses
# @param tokenizer -> the tokenizer used for processing text
# @param model -> the model used to generate embeddings
# @param k -> the number of top similar responses to return
# @return a list of the most similar responses based on cosine similarity
def find_most_similar_responses(question_embedding, responses, response_embeddings, tokenizer, model, k):
    
    similarities = [(response, get_cosine_similarity(question_embedding, response_embedding)) for response, response_embedding in zip(responses, response_embeddings)]
    # Sort the responses by similarity in descending order
    sorted_responses = sorted(similarities, key=lambda x: x[1], reverse=True)
    # Extract the top k most similar responses
    top_k_responses = [response[0] for response in sorted_responses[:k]]
    return top_k_responses


# @brief Finds the most similar questions to a given response based on cosine similarity.
# @param response_embedding -> the embedding of the response
# @param questions -> a list of question texts
# @param questions_embedding -> a list of embeddings corresponding to the questions
# @param tokenizer -> the tokenizer used for processing text
# @param model -> the model used to generate embeddings
# @param k -> the number of top similar questions to return
# @return a list of the most similar questions based on cosine similarity
def find_most_similar_questions(response_embedding, questions, questions_embedding, tokenizer, model, k):
    
    similarities = [(question, get_cosine_similarity(response_embedding, questions_embedding)) for question, questions_embedding in zip(questions, questions_embedding)]
    
    sorted_questions = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_k_questions = [question[0] for question in sorted_questions[:k]]
    return top_k_questions

# @brief Evaluates the accuracy of human responses in relation to predicted responses based on embeddings.
# @param data -> a DataFrame containing questions and their corresponding true human responses
# @param tokenizer -> the tokenizer used for processing text
# @param model -> the model used to generate embeddings
# @param embedding_cache -> a cache for storing precomputed embeddings
# @param response_embeddings -> a list of embeddings for known responses
# @return a tuple of total top-1 and top-5 accuracy scores
def evaluate_human_responses(data, tokenizer, model, embedding_cache, response_embeddings):
    total_top1_accuracy = 0
    total_top5_accuracy = 0

    i = 0
    for _, row in data.iterrows():
        question = row["soru"]
        true_human_response = row["insan cevabı"]

        # En benzer cevapları bulun
        five_most_similar_responses = find_most_similar_responses(get_embedding(question, tokenizer, model, embedding_cache), human_responses, response_embeddings, tokenizer, model, k=5)

        # Başarıyı hesaplayın
        top1_accuracy = 1 if true_human_response in five_most_similar_responses[:1] else 0
        top5_accuracy = 1 if true_human_response in five_most_similar_responses else 0

        # print(f"\nQuestion{i}: ", question)
        # print("True Response: ", true_human_response)
        # j = 1
        # for resp in five_most_similar_responses:
        #   print(j, ": ", resp)
        #   j += 1
        # i += 1
        total_top1_accuracy = total_top1_accuracy + top1_accuracy
        total_top5_accuracy = total_top5_accuracy + top5_accuracy

    return total_top1_accuracy, total_top5_accuracy

# @brief Evaluates the accuracy of machine responses in relation to predicted responses based on embeddings.
# @param data -> a DataFrame containing questions and their corresponding true machine responses
# @param tokenizer -> the tokenizer used for processing text
# @param model -> the model used to generate embeddings
# @param embedding_cache -> a cache for storing precomputed embeddings
# @param response_embeddings -> a list of embeddings for known responses
# @return a tuple of total top-1 and top-5 accuracy scores
def evaluate_machine_responses(data, tokenizer, model, embedding_cache, response_embeddings):
    total_top1_accuracy = 0
    total_top5_accuracy = 0

    i = 0
    for _, row in data.iterrows():
        question = row["soru"]
        true_machine_response = row["makine cevabı"]

  
        five_most_similar_responses = find_most_similar_responses(get_embedding(question, tokenizer, model, embedding_cache), machine_responses, response_embeddings, tokenizer, model, k=5)

        top1_accuracy = 1 if true_machine_response in five_most_similar_responses[:1] else 0
        top5_accuracy = 1 if true_machine_response in five_most_similar_responses else 0

        # print(f"\nQuestion{i}: ", question)
        # print("True Response: ", true_machine_response)
        # j = 1
        # for resp in five_most_similar_responses:
        #   print(j, ": ", resp)
        #   j += 1
        # i += 1
        total_top1_accuracy = total_top1_accuracy + top1_accuracy
        total_top5_accuracy = total_top5_accuracy + top5_accuracy

    return total_top1_accuracy, total_top5_accuracy

# @brief Evaluates the accuracy of finding the correct question from a given human response based on embeddings.
# @param data -> a DataFrame containing responses and their corresponding true questions
# @param tokenizer -> the tokenizer used for processing text
# @param model -> the model used to generate embeddings
# @param embedding_cache -> a cache for storing precomputed embeddings
# @param question_embeddings -> a list of embeddings for known questions
# @return a tuple of total top-1 and top-5 accuracy scores
def evaluate_questions_from_human_response(data, tokenizer, model, embedding_cache, question_embeddings):
    total_top1_accuracy = 0
    total_top5_accuracy = 0

    i = 0
    for _, row in data.iterrows():
        response  = row["insan cevabı"]
        true_question = row["soru"]

        five_most_similar_questions = find_most_similar_questions(get_embedding(response, tokenizer, model, embedding_cache), questions, question_embeddings, tokenizer, model, k=5)

        top1_accuracy = 1 if true_question in five_most_similar_questions[:1] else 0
        top5_accuracy = 1 if true_question in five_most_similar_questions else 0

        # print(f"\nResponse{i}: ", response)
        # print("True Question: ", true_question)
        # j = 1
        # for question in five_most_similar_questions:
        #   print(j, ": ", question)
        #   j += 1
        # i += 1
        total_top1_accuracy = total_top1_accuracy + top1_accuracy
        total_top5_accuracy = total_top5_accuracy + top5_accuracy

    return total_top1_accuracy, total_top5_accuracy

# @brief Evaluates the accuracy of finding the correct question from a given machine response based on embeddings.
# @param data -> a DataFrame containing responses and their corresponding true questions
# @param tokenizer -> the tokenizer used for processing text
# @param model -> the model used to generate embeddings
# @param embedding_cache -> a cache for storing precomputed embeddings
# @param question_embeddings -> a list of embeddings for known questions
# @return a tuple of total top-1 and top-5 accuracy scores
def evaluate_questions_from_machine_response(data, tokenizer, model, embedding_cache, question_embeddings):
    total_top1_accuracy = 0
    total_top5_accuracy = 0

    i = 0
    for _, row in data.iterrows():
        response  = row["makine cevabı"]
        true_question = row["soru"]

        five_most_similar_questions = find_most_similar_questions(get_embedding(response, tokenizer, model, embedding_cache), questions, question_embeddings, tokenizer, model, k=5)

        top1_accuracy = 1 if true_question in five_most_similar_questions[:1] else 0
        top5_accuracy = 1 if true_question in five_most_similar_questions else 0

        # print(f"\nResponse{i}: ", response)
        # print("True Question: ", true_question)
        # j = 1
        # for question in five_most_similar_questions:
        #   print(j, ": ", question)
        #   j += 1
        # i += 1
        total_top1_accuracy = total_top1_accuracy + top1_accuracy
        total_top5_accuracy = total_top5_accuracy + top5_accuracy

    return total_top1_accuracy, total_top5_accuracy

startTime = time.time()
bert_total_top1_accuracy, bert_total_top5_accuracy = evaluate_human_responses(data, tokenizer_bert, model_bert, bert_embedding_cache, bert_human_response_embeddings)
print("Bert Question To Human Response")
print("Total Top-1 Accuracy:", bert_total_top1_accuracy)
print("Total Top-5 Accuracy:", bert_total_top5_accuracy)
endTime = time.time()
print("Zaman: ", endTime - startTime)

startTime = time.time()
bert_total_top1_accuracy, bert_total_top5_accuracy = evaluate_machine_responses(data, tokenizer_bert, model_bert, bert_embedding_cache, bert_machine_response_embeddings)
print("Bert Question To Machine Response")
print("Total Top-1 Accuracy:", bert_total_top1_accuracy)
print("Total Top-5 Accuracy:", bert_total_top5_accuracy)
endTime = time.time()
print("Zaman: ", endTime - startTime)

startTime = time.time()
bert_total_top1_accuracy, bert_total_top5_accuracy = evaluate_questions_from_human_response(data, tokenizer_bert, model_bert, bert_embedding_cache, bert_question_embeddings)
print("Bert Human Response To Question")
print("Total Top-1 Accuracy:", bert_total_top1_accuracy)
print("Total Top-5 Accuracy:", bert_total_top5_accuracy)
endTime = time.time()
print("Zaman: ", endTime - startTime)

startTime = time.time()
bert_total_top1_accuracy, bert_total_top5_accuracy = evaluate_questions_from_machine_response(data, tokenizer_bert, model_bert, bert_embedding_cache, bert_question_embeddings)
print("Bert Machine Response To Question")
print("Total Top-1 Accuracy:", bert_total_top1_accuracy)
print("Total Top-5 Accuracy:", bert_total_top5_accuracy)
endTime = time.time()
print("Zaman: ", endTime - startTime)

# startTime = time.time()
# gpt2_total_top1_accuracy, gpt2_total_top5_accuracy = evaluate_human_responses(data, tokenizer_gpt2, model_gpt2, gpt2_embedding_cache, gpt2_human_response_embeddings)
# print("GPT2 Question To Human Response")
# print("Total Top-1 Accuracy:", gpt2_total_top1_accuracy)
# print("Total Top-5 Accuracy:", gpt2_total_top5_accuracy)
# endTime = time.time()
# print("Zaman: ", endTime - startTime)

# startTime = time.time()
# gpt2_total_top1_accuracy, gpt2_total_top5_accuracy = evaluate_machine_responses(data, tokenizer_gpt2, model_gpt2, gpt2_embedding_cache, gpt2_machine_response_embeddings)
# print("GPT2 Question To Machine Response")
# print("Total Top-1 Accuracy:", gpt2_total_top1_accuracy)
# print("Total Top-5 Accuracy:", gpt2_total_top5_accuracy)
# endTime = time.time()
# print("Zaman: ", endTime - startTime)

# startTime = time.time()
# gpt2_total_top1_accuracy, gpt2_total_top5_accuracy = evaluate_questions_from_human_response(data, tokenizer_gpt2, model_gpt2, gpt2_embedding_cache, gpt2_question_embeddings)
# print("GPT2 Human Response To Question")
# print("Total Top-1 Accuracy:", gpt2_total_top1_accuracy)
# print("Total Top-5 Accuracy:", gpt2_total_top5_accuracy)
# endTime = time.time()
# print("Zaman: ", endTime - startTime)

# startTime = time.time()
# gpt2_total_top1_accuracy, gpt2_total_top5_accuracy = evaluate_questions_from_machine_response(data, tokenizer_gpt2, model_gpt2, gpt2_embedding_cache, gpt2_question_embeddings)
# print("GPT2 Machine Response To Question")
# print("Total Top-1 Accuracy:", gpt2_total_top1_accuracy)
# print("Total Top-5 Accuracy:", gpt2_total_top5_accuracy)
# endTime = time.time()
# print("Zaman: ", endTime - startTime)

# Bert TSNE
combined_embeddings = np.concatenate((bert_question_embeddings, bert_human_response_embeddings, bert_machine_response_embeddings), axis=0)

tsne = TSNE(n_components=2, random_state=42)

tsne_embeddings = tsne.fit_transform(combined_embeddings)

tsne_df = pd.DataFrame(tsne_embeddings, columns=['X', 'Y'])

tsne_df['Type'] = ['Soru'] * len(bert_question_embeddings) + ['İnsan Cevabı'] * len(bert_human_response_embeddings) + ['Makine Cevabı'] * len(bert_machine_response_embeddings)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='X', y='Y', hue='Type', data=tsne_df)
plt.title('Bert TSNE Görselleştirmesi')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.show()

# # GPT TSNE
# combined_embeddings = np.concatenate((gpt2_question_embeddings, gpt2_human_response_embeddings, gpt2_machine_response_embeddings), axis=0)

# tsne = TSNE(n_components=2, random_state=42)

# tsne_embeddings = tsne.fit_transform(combined_embeddings)

# tsne_df = pd.DataFrame(tsne_embeddings, columns=['X', 'Y'])

# tsne_df['Type'] = ['Soru'] * len(gpt2_question_embeddings) + ['İnsan Cevabı'] * len(gpt2_human_response_embeddings) + ['Makine Cevabı'] * len(gpt2_machine_response_embeddings)

# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='X', y='Y', hue='Type', data=tsne_df)
# plt.title('GPT2 TSNE Görselleştirmesi')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend(loc='best')
# plt.show()