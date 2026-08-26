import torch

from src.backbone.ddnm_projection import ddnm_project


def _average_pool(x: torch.Tensor, scale: int) -> torch.Tensor:
    c, h, w = x.shape
    return x.view(c, h // scale, scale, w // scale, scale).mean(dim=(2, 4))


def test_ddnm_project_enforces_exact_data_consistency():
    torch.manual_seed(0)
    scale = 4
    x_hat = torch.rand(3, 16, 16)
    y = torch.rand(3, 4, 4)  # arbitrary target observation, unrelated to x_hat
    x_proj = ddnm_project(x_hat, y, scale)
    reprojected = _average_pool(x_proj, scale)
    assert torch.allclose(reprojected, y, atol=1e-5)


def test_ddnm_project_preserves_null_space_detail():
    torch.manual_seed(1)
    scale = 4
    x_hat = torch.rand(3, 16, 16)
    y = _average_pool(x_hat, scale)  # already consistent
    x_proj = ddnm_project(x_hat, y, scale)
    assert torch.allclose(x_proj, x_hat, atol=1e-5)


def test_ddnm_project_output_shape():
    x_hat = torch.rand(3, 32, 32)
    y = torch.rand(3, 8, 8)
    x_proj = ddnm_project(x_hat, y, scale=4)
    assert x_proj.shape == x_hat.shape


def test_ddnm_project_preserves_null_space_detail_with_mismatched_y():
    torch.manual_seed(2)
    scale = 4
    x_hat = torch.rand(3, 16, 16)
    y = torch.rand(3, 4, 4)  # arbitrary, mismatched with x_hat's own block means

    def _within_block_detail(x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape
        blocks = x.reshape(c, h // scale, scale, w // scale, scale)
        return blocks - blocks.mean(dim=(2, 4), keepdim=True)

    x_proj = ddnm_project(x_hat, y, scale)
    assert torch.allclose(_within_block_detail(x_proj), _within_block_detail(x_hat), atol=1e-5)


def test_ddnm_project_handles_non_contiguous_input():
    torch.manual_seed(3)
    scale = 4
    x_hat = torch.rand(16, 16, 3).permute(2, 0, 1)  # (H, W, C) -> (C, H, W), non-contiguous
    assert not x_hat.is_contiguous()
    y = torch.rand(3, 4, 4)
    x_proj = ddnm_project(x_hat, y, scale)
    reprojected = _average_pool(x_proj, scale)
    assert torch.allclose(reprojected, y, atol=1e-5)


def test_ddnm_project_scale_one_is_identity_on_y():
    x_hat = torch.rand(3, 8, 8)
    y = torch.rand(3, 8, 8)
    x_proj = ddnm_project(x_hat, y, scale=1)
    assert torch.allclose(x_proj, y, atol=1e-5)
