from flare_ai_consensus.embeddings import EmbeddingModel

from math import factorial
import re

import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations
from collections import Counter
import scipy.spatial.distance as distance
from sklearn.metrics.pairwise import cosine_similarity


def _concatenate_aggregator(responses: dict[str, str]) -> str:
    """
    Aggregate responses by concatenating each model's answer with a label.

    :param responses: A dictionary mapping model IDs to their response texts.
    :return: A single aggregated string.
    """
    return "\n\n".join([f"{model}: {text}" for model, text in responses.items()])


async def async_centralized_embedding_aggregator(
        embedding_model: EmbeddingModel,
        responses: dict[str, str],
):
    if not responses:
        return ""

    if len(responses) == 1:
        # If there's only one response, return it directly
        return list(responses.values())[0]

    # Get embeddings for each response
    embeddings_dict = await _get_embeddings_for_responses(embedding_model, responses)

    # Find the response closest to the center
    closest_model_id = find_best_embedding(embeddings_dict)
    shapley_values, main_cluster_models = calculate_shapley_values(embeddings_dict)

    # Calculate confidence scores
    confidence_score = await calculate_confidence_scores(embedding_model, responses, main_cluster_models)
    return responses[closest_model_id], shapley_values, confidence_score


def calculate_shapley_values(embeddings_dict: dict[str, np.ndarray]):
    """
    Calculate Shapley values for each model based on their embeddings, using DBSCAN
    clustering to identify the main cluster and outliers.

    :param embeddings_dict: A dictionary mapping model IDs to their embedding vectors.
    :return: A dictionary mapping model IDs to their Shapley values.
    """
    # Extract model IDs and embeddings
    models = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[model] for model in models])
    N = len(models)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=max(2, int(N * 0.1)))
    cluster_labels = dbscan.fit_predict(embeddings)

    # Identify the main cluster (largest cluster)
    cluster_counts = Counter(cluster_labels)
    if -1 in cluster_counts:
        del cluster_counts[-1]

    # If all points are noise or no clusters found
    if not cluster_counts:
        # Assign equal Shapley values if no clear clusters
        return {model: 1.0 / N for model in models}, models

    # Get the largest cluster
    main_cluster_label = cluster_counts.most_common(1)[0][0]

    # print models in main cluster
    main_cluster_indices = [i for i, label in enumerate(cluster_labels) if label == main_cluster_label]
    main_cluster_models = [models[i] for i in main_cluster_indices]
    print("Models in main cluster:", main_cluster_models)

    def utility(coalition):
        if not coalition:
            return 0.0

        # Extract embeddings for this coalition
        coalition_embeddings = np.array([embeddings_dict[model] for model in coalition])

        # If only one model in coalition, use a different approach
        if len(coalition) == 1:
            model_idx = models.index(coalition[0])
            # Higher utility if in main cluster, lower otherwise
            return 2.0 if cluster_labels[model_idx] == main_cluster_label else 0.5

        # For multiple models, calculate centroid
        centroid = np.mean(coalition_embeddings, axis=0)

        # Calculate distances from each model to centroid
        distances = [distance.cosine(centroid, embeddings_dict[model]) for model in coalition]

        # Count models in main cluster
        main_cluster_count = sum(1 for i, model in enumerate(coalition)
                                 if cluster_labels[models.index(model)] == main_cluster_label)

        # Higher utility if more models from main cluster and lower average distance
        coherence = 1.0 / (1.0 + np.mean(distances))
        main_cluster_ratio = main_cluster_count / len(coalition)

        return coherence * (1 + main_cluster_ratio)

    # Step 4: Calculate Shapley values
    shapley_values = {model: 0.0 for model in models}

    # For each model, calculate its marginal contribution across all possible coalitions
    for model in models:
        other_models = [m for m in models if m != model]

        # Iterate through all possible coalition sizes
        for size in range(len(other_models) + 1):
            # Generate all possible coalitions of the given size
            for coalition in combinations(other_models, size):
                coalition = list(coalition)

                # Calculate marginal contribution
                with_model = utility(coalition + [model])
                without_model = utility(coalition)
                marginal_contribution = with_model - without_model

                # Weight by coalition size
                weight = (factorial(size) * factorial(N - size - 1)) / factorial(N)
                shapley_values[model] += weight * marginal_contribution

    # Normalize Shapley values to sum to 1
    total_value = sum(shapley_values.values())
    if total_value > 0:
        shapley_values = {model: value / total_value for model, value in shapley_values.items()}

    print("Shapley values:", shapley_values)
    return shapley_values, main_cluster_models


async def calculate_confidence_scores(
        embedding_model: EmbeddingModel,
        responses: dict[str, str],
        main_cluster_models: list[str]
):
    valid_responses = {model: response for model, response in responses.items() if model in main_cluster_models}

    operations = [extract_values(response) for response in valid_responses.values()]

    operation_counts = Counter([op["operation"] for op in operations])
    token_a_counts = Counter([op["token_a"] for op in operations])
    token_b_counts = Counter([op["token_b"] for op in operations])
    amounts = [op["amount"] for op in operations]
    embeddings = [await _get_embedding(embedding_model, op["reason"]) for op in operations]

    cosine_similarities = cosine_similarity(embeddings)
    upper_triangle_values = cosine_similarities[np.triu_indices(len(embeddings), k=1)]
    mean_cosine_similarity = np.mean(upper_triangle_values)
    std_cosine_similarity = np.std(upper_triangle_values)
    cosine_confidence = 1 - (std_cosine_similarity / mean_cosine_similarity)

    # Swap/Lend Confidence
    total_operations = len(operations)
    swap_confidence = operation_counts["SWAP"] / total_operations
    lend_confidence = operation_counts["LEND"] / total_operations

    # Token Confidence
    token_a_confidence = sum(token_a_counts.values()) / total_operations
    token_b_confidence = sum(token_b_counts.values()) / total_operations

    # Amount Confidence (mean and variability)
    avg_amount = np.mean(amounts)
    std_amount = np.std(amounts)
    amount_confidence = 1 - (std_amount / avg_amount)

    w_cosine = 0.25
    w_swap = 0.2
    w_lend = 0.2
    w_token_a = 0.1
    w_token_b = 0.1
    w_amount = 0.15

    overall_confidence = (w_cosine * cosine_confidence) + \
                         (w_swap * swap_confidence) + \
                         (w_lend * lend_confidence) + \
                         (w_token_a * token_a_confidence) + \
                         (w_token_b * token_b_confidence) + \
                         (w_amount * amount_confidence)

    return overall_confidence


def extract_values(text) -> dict[str, str]:
    pattern = r'\{["\']operation["\']: ["\'](.*?)["\'], ["\']token_a["\']: ["\'](.*?)["\'], ["\']token_b["\']: ["\'](.*?)["\'], ["\']amount["\']: ["\'](.*?)["\'], ["\']reason["\']: ["\'](.*?)["\']\}'
    match = re.search(pattern, text)

    if match:
        return {
            "operation": match.group(1),
            "token_a": match.group(2),
            "token_b": match.group(3),
            "amount": float(match.group(4)),
            "reason": match.group(5),
        }
    return {
        "operation": "",
        "token_a": "",
        "token_b": "",
        "amount": float(0),
        "reason": "",
    }


def find_best_embedding(embeddings_dict, eps=0.2, min_samples=2):
    # Extract model IDs and embeddings
    models = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[model] for model in models])

    # Normalize embeddings for better clustering
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(normalized_embeddings)

    # Get main cluster (largest cluster)
    cluster_counts = Counter(cluster_labels)
    if -1 in cluster_counts:  # Remove noise points
        del cluster_counts[-1]

    # Handle case where no clusters were found
    if not cluster_counts:
        # Fall back to the model closest to the global mean
        global_mean = np.mean(embeddings, axis=0)
        best_model = min(models, key=lambda m: np.linalg.norm(embeddings_dict[m] - global_mean))
        return best_model

    main_cluster_label = cluster_counts.most_common(1)[0][0]

    # Get models in main cluster
    main_cluster_indices = [i for i, label in enumerate(cluster_labels)
                            if label == main_cluster_label]
    main_cluster_models = [models[i] for i in main_cluster_indices]

    # Find model with minimum sum of distances to all other models in cluster
    best_model = None
    min_total_distance = float('inf')

    for model1 in main_cluster_models:
        total_distance = 0
        for model2 in main_cluster_models:
            if model1 != model2:
                dist = np.linalg.norm(embeddings_dict[model1] - embeddings_dict[model2])
                total_distance += dist

        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_model = model1

    return best_model


async def _get_embeddings_for_responses(
        embedding_model: EmbeddingModel, responses: dict[str, str]
) -> dict[str, np.ndarray]:
    """
    Get embeddings for each response using the provider's embedding API.

    :param provider: An asynchronous OpenRouterProvider.
    :param responses: A dictionary mapping model IDs to their response texts.
    :return: A dictionary mapping model IDs to their embedding vectors.
    """
    embeddings = {}
    for model_id, text in responses.items():
        text_ = await embedding_model.get_embeddings(text)
        embeddings[model_id] = np.array(text_.embeddings[0].values)
    return embeddings


async def _get_embedding(
        embedding_model: EmbeddingModel, text: str
) -> np.ndarray:
    """
    Get embeddings for a given text using the provider's embedding API.

    :param provider: An asynchronous OpenRouterProvider.
    :param text: The text to embed.
    :return: The embedding vector.
    """
    text_ = await embedding_model.get_embeddings(text)
    return np.array(text_.embeddings[0].values)
