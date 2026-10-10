"""Feature builders shared by all arms: count vectors, TF-IDF, template embeddings."""
import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

from dataset_paths import processed_path


def load_templates(dataset, selftest):
    return pd.read_parquet(processed_path(dataset, "templates.parquet", selftest))


def vocab_size(dataset, selftest):
    return len(load_templates(dataset, selftest))


def count_matrix(sessions, n_vocab):
    """sessions: DataFrame with a 'template_seq' column of lists of template ids."""
    rows, cols, data = [], [], []
    for i, seq in enumerate(sessions["template_seq"]):
        if len(seq) == 0:
            continue
        ids, counts = np.unique(seq, return_counts=True)
        rows.extend([i] * len(ids))
        cols.extend(ids.tolist())
        data.extend(counts.tolist())
    data = np.asarray(data, dtype="float32")
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(sessions), n_vocab))


def tfidf_matrix(counts, fitted_transformer=None):
    if fitted_transformer is None:
        fitted_transformer = TfidfTransformer()
        mat = fitted_transformer.fit_transform(counts)
        return mat, fitted_transformer
    return fitted_transformer.transform(counts), fitted_transformer


_EMBED_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBED_MODEL


def template_embeddings(dataset, selftest):
    """Cached per-template embedding matrix, shape (n_vocab, 384), row i = template_id i."""
    cache = processed_path(dataset, "template_embeddings.npy", selftest)
    if cache.exists():
        return np.load(cache)
    templates = load_templates(dataset, selftest).sort_values("template_id")
    model = _get_embed_model()
    emb = model.encode(templates["template"].tolist(), show_progress_bar=False, batch_size=64)
    emb = np.asarray(emb, dtype="float32")
    np.save(cache, emb)
    return emb


def session_embedding_matrix(sessions, template_emb):
    """Count-weighted mean of template embeddings per session. Returns (n_sessions, dim)."""
    dim = template_emb.shape[1]
    out = np.zeros((len(sessions), dim), dtype="float32")
    for i, seq in enumerate(sessions["template_seq"]):
        if len(seq) == 0:
            continue
        ids, counts = np.unique(seq, return_counts=True)
        vecs = template_emb[ids]
        out[i] = np.average(vecs, axis=0, weights=counts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["HDFS_v1", "BGL"], required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    sessions = pd.read_parquet(processed_path(args.dataset, "sessions.parquet", args.selftest))
    n_vocab = vocab_size(args.dataset, args.selftest)
    counts = count_matrix(sessions, n_vocab)
    tfidf, _ = tfidf_matrix(counts)
    emb = template_embeddings(args.dataset, args.selftest)
    sess_emb = session_embedding_matrix(sessions, emb)
    print(f"{args.dataset} selftest={args.selftest}: n_sessions={len(sessions)} n_vocab={n_vocab} "
          f"counts={counts.shape} tfidf={tfidf.shape} template_emb={emb.shape} session_emb={sess_emb.shape}")


if __name__ == "__main__":
    main()
