import numpy as np
import pytest
import torch

import helpers.models
from transformers import AutoModel, AutoTokenizer

from helpers.models import (
    EsmMeanPoolClassifier,
    cached_embeddings,
    embed_sequences,
    loss_history_frame,
    make_trainer,
    pick_device,
    predict_classes,
)

pytestmark = pytest.mark.model

ENO1 = "MSILKIHAREIFDSRGNPTVEVDLFTSKGLFRAAVPSGASTGIYEALELRDNDKTRYMGKGVSKAVEHINKTIAPALVSKKLNVTEQEKIDKLMIEMDGTENKSKFGANAILGVSLAVCKAGAVEKGVPLYRHIADLAGNSEVILPVPAFNVINGGSHAGNKLAMQEFMILPVGAANFREAMRIGAEVYHNLKNVIKEKYGKDATNVGDEGGFAPNILENKEGLELLKTAIGKAGYTDKVVIGMDVAASEFFRSGKYDLDFKSPDDPSRYISPDQLADLYKSFIKDYPVVSIEDPFDQDDWGAWQKFTASAGIQVVGDDLTVTNPKRIAKAVNEKSCNCLLLKVNQIGSVTESLQACKLAQANGWGVMVSHRSGETEDTFIADLVVGLCTGQIKTGAPCRSERLAKYNQLLRIEEELGSKAKFAGRNFRNPLAK"
SHORT = "MKTAYIAKQR"


def test_pick_device():
    assert pick_device() in {"cuda", "mps", "cpu"}


def test_embed_shape():
    out = embed_sequences([SHORT, ENO1])
    assert out.shape == (2, 320) and out.dtype == np.float32


def test_embed_padding_and_order():
    alone = embed_sequences([SHORT], batch_size=1)
    batched = embed_sequences([ENO1, SHORT], batch_size=16)
    np.testing.assert_allclose(alone[0], batched[1], atol=1e-4)


@pytest.fixture(scope="session")
def trained_tiny(tmp_path_factory, tiny_train_val):
    d = tmp_path_factory.mktemp("tiny")
    tr = make_trainer(*tiny_train_val, output_dir=d / "ck", num_epochs=1, max_length=64, batch_size=4)
    tr.train()
    tr.save_model(d / "m")
    return tr, d / "m", tiny_train_val[1]


def test_make_trainer_tiny(trained_tiny):
    tr, model_dir, val = trained_tiny
    assert tr.args.fp16 == torch.cuda.is_available()
    h = loss_history_frame(tr)
    assert list(h.columns) == ["step", "epoch", "train_loss", "eval_loss", "eval_macro_f1"]
    assert h.eval_loss.notna().any()
    pred = predict_classes(model_dir, list(val.sequence))
    assert set(pred) <= set(range(1, 8)) and len(pred) == len(val)
    assert embed_sequences(list(val.sequence), str(model_dir)).shape[1] == 320


def test_long_and_unusual_residues(trained_tiny):
    seq = ("ACDEFGHIKLMNPQRSTVWYUXO" * 70)[:1500]
    assert np.isfinite(embed_sequences([seq])).all()
    assert predict_classes(trained_tiny[1], [seq])[0] in range(1, 8)


def test_cached_embeddings_recomputes_on_mismatch(tmp_path, monkeypatch, tiny_df):
    calls = []
    real = helpers.models.embed_sequences
    monkeypatch.setattr(helpers.models, "embed_sequences", lambda s, *a, **k: calls.append(len(s)) or real(s, *a, **k))
    p = tmp_path / "e.npz"
    cached_embeddings(tiny_df, p)
    cached_embeddings(tiny_df, p)
    assert len(calls) == 1
    cached_embeddings(tiny_df.iloc[::-1], p)
    assert len(calls) == 2


def _class_predictions(model, model_dir, sequences):
    """Predicted classes (1-7) from an in-memory model, one batch, in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    batch = tokenizer(sequences, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model.eval()(**batch).logits
    return logits.argmax(dim=1).cpu().numpy() + 1


def test_mean_pool_matches_embeddings(trained_tiny):
    _, model_dir, val = trained_tiny
    encoder, info = AutoModel.from_pretrained(model_dir, add_pooling_layer=False, output_loading_info=True)
    assert not info["missing_keys"]  # every encoder weight comes from the save; no classifier needed
    model = EsmMeanPoolClassifier.from_pretrained(model_dir).eval()
    embeddings = torch.from_numpy(embed_sequences(list(val.sequence), str(model_dir)))
    with torch.inference_mode():
        from_embeddings = model.classifier(embeddings).argmax(dim=1).numpy() + 1
    np.testing.assert_array_equal(from_embeddings, predict_classes(model_dir, list(val.sequence)))


def test_mean_pool_ignores_padding(trained_tiny):
    _, model_dir, _ = trained_tiny
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = EsmMeanPoolClassifier.from_pretrained(model_dir).eval()
    with torch.inference_mode():
        alone = model(**tokenizer([SHORT], return_tensors="pt")).logits
        batched = model(**tokenizer([SHORT, ENO1], padding=True, return_tensors="pt")).logits
    np.testing.assert_allclose(alone[0].numpy(), batched[0].numpy(), atol=1e-4)


def test_saved_model_reloads(trained_tiny):
    tr, model_dir, val = trained_tiny
    seqs = list(val.sequence)
    before = _class_predictions(tr.model, model_dir, seqs)
    reloaded = EsmMeanPoolClassifier.from_pretrained(model_dir).to(tr.model.device)
    assert isinstance(tr.model, EsmMeanPoolClassifier)
    np.testing.assert_array_equal(_class_predictions(reloaded, model_dir, seqs), before)
