from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def cross_encoder_score(self, query, contexts):
        cross_encoder = self.cross_encoder
        pairs = [(query, context) for context in contexts]
        scores = cross_encoder.predict(pairs)
        # context_with_score_list = [(score, context) for score, context in zip(scores, contexts)]
        # return context_with_score_list.sort(key=lambda x: x[1], reverse=True)

        return scores

    def source_reliability_score(self, source_type):
        reliability_weight = {
            "vector_search": 0.8,
            "graph_search": 0.9,
            "web_search": 0.6,
        }

        return reliability_weight.get(source_type, 0.5)

    def rerank(self, query, contexts):
        weights = {
            "cross_encoder_score": 0.8,
            # "source_reliability_score": 0.2
        }
        scores = []
        for cxt in contexts:
            ce_score = self.cross_encoder_score(query, cxt)
            # sr_score = self.source_reliability_score(cxt["tool_name"])

            # final_score = weights.get("cross_encoder_score", 0.5) * ce_score + weights.get("source_reliability_score", 0.5) * sr_score
            # scores.append(final_score)

            scores.append(ce_score)
        context_with_score_list = [
            (score, context) for score, context in zip(scores, contexts)
        ]
        return context_with_score_list.sort(key=lambda x: x[1], reverse=True)
