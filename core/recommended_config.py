"""
To achieve a good quality-diversity trade-off, we provide recommended model configurations
for different image resolutions. Firstly, it is recommended to have the input noise dimensions
between 3 and 7. Secondly, it is advisable to keep the layout representation dims roughly as 32x32.
"""
recommended_noise_range = (3, 7)
recommended_blocks_by_res = {
# G_blocks - D blocks - D_low-level blocks
9: (8, 4), # ~1024
8: (7, 4), # ~512
7: (7, 3), # ~256
6: (7, 3), # ~128
5: (6, 2), # ~64
}


def get_recommended_config(orig_res):
    """
    Given the resolution of original training images, this function produces
    the recommended model configuration and the training image resolution.
    """
    ans = dict()
    for num_g in recommended_blocks_by_res:
        multiplier = 2 ** (num_g - 1)
        ok_1 = recommended_noise_range[0] <= orig_res[0] / multiplier <= recommended_noise_range[1]
        ok_2 = recommended_noise_range[0] <= orig_res[1] / multiplier <= recommended_noise_range[1]
        ans[num_g] = (ok_1, ok_2)

    ok_1 = ok_2 = False
    for num_g in sorted(ans):
        if ans[num_g][0]:
            ok_1 = True
        if ans[num_g][1]:
            ok_2 = True
        if ok_1 and ok_2:
            recommended_G = num_g
            break

    noise_shape = (round(orig_res[0] / 2 ** (recommended_G - 1)), round(orig_res[1] / 2 ** (recommended_G - 1)))
    resolution = (noise_shape[0] * 2 ** (recommended_G - 1), noise_shape[1] * 2 ** (recommended_G - 1))
    recommended_D, recommended_D0  = recommended_blocks_by_res[recommended_G]
    return resolution, [noise_shape, recommended_G, recommended_D, recommended_D0]

