import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

class FourierDomainAdaptation(ImageOnlyTransform):
    def __init__(self, alpha=0.1, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.alpha = alpha  

    def apply(self, img, mix_img=None, **params):
        
        # converting images to the frequency domain
        fft_img = np.fft.fft2(img, axes=(0, 1))
        fft_mix = np.fft.fft2(mix_img, axes=(0, 1))

        fft_img_shift = np.fft.fftshift(fft_img, axes=(0, 1))
        fft_mix_shift = np.fft.fftshift(fft_mix, axes=(0, 1))

        amp_img, phase_img = np.abs(fft_img_shift), np.angle(fft_img_shift)
        amp_mix = np.abs(fft_mix_shift)

        mixed_amp = (1 - self.alpha) * amp_img + self.alpha * amp_mix

        mixed_fft = np.fft.ifftshift(mixed_amp * np.exp(1j * phase_img), axes=(0, 1))
        mixed_img = np.fft.ifft2(mixed_fft, axes=(0, 1)).real

        return np.clip(mixed_img, 0, 255).astype(np.uint8)

    def get_params_dependent_on_targets(self, params):
        return {"mix_img": params["image2"]}

    @property
    def targets_as_params(self):
        return ["image2"]
