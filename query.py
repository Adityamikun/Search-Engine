import math

def load_vocab():
    vocab = {}
    with open('tf-idf/vocab.txt', 'r', encoding='latin-1') as f:
        vocab_terms = f.readlines()
    with open('tf-idf/idf-values.txt', 'r', encoding='latin-1') as f:
        idf_values = f.readlines()

    for (term, idf_value) in zip(vocab_terms, idf_values):
        vocab[term.strip()] = float(idf_value.strip())

    return vocab

def load_documents():
    documents = []
    with open('tf-idf/documents.txt', 'r', encoding='latin-1') as f:
        documents = f.readlines()
    documents = [document.strip().split() for document in documents]

    print('Number of documents:', len(documents))
    print('Sample document:', documents[0])
    return documents

def load_inverted_index():
    inverted_index = {}
    with open('tf-idf/inverted-index.txt', 'r', encoding='latin-1') as f:
        inverted_index_terms = f.readlines()

    for row_num in range(0, len(inverted_index_terms), 2):
        term = inverted_index_terms[row_num].strip()
        documents = inverted_index_terms[row_num + 1].strip().split()
        inverted_index[term] = documents

    print('Size of inverted index:', len(inverted_index))
    return inverted_index

def load_question_links():
    question_links = []
    with open('Question data/Qindex.txt', 'r', encoding='latin-1') as f:
        question_links = f.readlines()
    question_links = [link.strip() for link in question_links]

    return question_links

def get_tf_dictionary(document):
    tf_values = {}
    total_terms = len(document)
    for term in document:
        if term not in tf_values:
            tf_values[term] = 1 / total_terms
        else:
            tf_values[term] += 1 / total_terms

    return tf_values

def get_idf_value(term, vocab_idf_values):
    return vocab_idf_values.get(term, 0)

def calculate_sorted_order_of_documents(query_terms, documents, vocab_idf_values, inverted_index):
    scores = {}
    for term in query_terms:
        if term in inverted_index:
            idf = get_idf_value(term, vocab_idf_values)
            tf_query = query_terms.count(term)
            tf_doc = get_tf_dictionary(query_terms)
            for doc in inverted_index[term]:
                if doc not in scores:
                    scores[doc] = 0
                scores[doc] += tf_query * tf_doc[term] * idf
    
    if not scores:
        return []  # Return an empty list if no matching questions found
    
    # Sort the documents based on the scores in descending order
    sorted_documents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    top_ten_links = sorted_documents[:10]  # Get the top ten documents
    
    # Convert document indices to question links
    question_links = load_question_links()
    top_ten_question_links = []
    for link in top_ten_links:
        question_index = int(link)
        if question_index < len(question_links):
            top_ten_question_links.append(question_links[question_index])
    
    return top_ten_question_links

query_string = input('Enter your query: ')
query_terms = [term.lower() for term in query_string.strip().split()]

vocab_idf_values = load_vocab()
documents = load_documents()
inverted_index = load_inverted_index()

potential_links = calculate_sorted_order_of_documents(query_terms, documents, vocab_idf_values, inverted_index)
if potential_links:
    for link in potential_links:
        print('Question Link:', link)
else:
    print('No matching questions found.')
