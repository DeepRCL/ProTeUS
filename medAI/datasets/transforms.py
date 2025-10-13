import torch
import torchvision.transforms.v2 as T

class PatchTransform:
    def __call__(self, patch):
        # expects numpy array [H,W] 0..255
        patch = torch.from_numpy(patch).float() / 255.0
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)
        return patch

class Identity:
    def __call__(self, *images):
        return images[0] if len(images) == 1 else images

class SpeckleNoise:
    def __call__(self, *images):
        outputs = []
        for image in images:
            C, H, W = image.shape
            gauss = torch.randn((C, H, W))
            noisy = image + 0.5 * image * gauss
            outputs.append(noisy)
        return outputs[0] if len(outputs) == 1 else outputs

class RandomTranslation:
    def __init__(self, translation=(0.1, 0.1)):
        self.translation = translation

    def __call__(self, *images):
        from torchvision.transforms.functional import affine
        from random import uniform

        h_factor, w_factor = uniform(-self.translation[0], self.translation[0]), \
                             uniform(-self.translation[1], self.translation[1])

        outputs = []
        for image in images:
            H, W = image.shape[-2:]
            translate_x = int(w_factor * W)
            translate_y = int(h_factor * H)
            outputs.append(
                affine(image, angle=0, translate=(translate_x, translate_y), scale=1, shear=0)
            )
        return outputs[0] if len(outputs) == 1 else outputs

class CorewiseTransform:
    def __call__(self, *images):
        transform = RandomTranslation()
        return transform(*images)

class RandomScaling:
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, *images):
        from torchvision.transforms.functional import affine
        from random import uniform
        scale_factor = uniform(self.scale_range[0], self.scale_range[1])
        outputs = [affine(img, angle=0, translate=(0, 0), scale=scale_factor, shear=0) for img in images]
        return outputs[0] if len(outputs) == 1 else outputs

class AugmentThruTime:
    def __call__(self, mode, *images):
        if mode == "weak":
            return WeakAugment()(*images)
        if mode == "strong":
            return StrongAugment()(*images)
        return images

class WeakAugment:
    def __init__(self):
        self.aug = T.RandomChoice([RandomTranslation()])

    def __call__(self, *images):
        outputs = []
        for image in images:
            outputs.append(self.aug(image))
        return outputs[0] if len(outputs) == 1 else outputs

class StrongAugment:
    def __init__(self):
        def h_linecut(img, p=0.2, size=128):
            im = img.clone()
            if torch.rand(1) > p: return im
            x = torch.randint(0, im.shape[1] - size, (1,))
            im[:, x:x+size, :] = 0
            return im

        def v_linecut(img, p=0.2, size=128):
            im = img.clone()
            if torch.rand(1) > p: return im
            y = torch.randint(0, im.shape[2] - size, (1,))
            im[:, :, y:y+size] = 0
            return im

        def pixel_cut(img, p=0.4, pixelp=0.03):
            im = img.clone()
            if torch.rand(1) > p: return im
            prob = torch.rand_like(im)
            im[prob < pixelp] = 0
            return im

        def pixel_saturation(img, p=0.4, pixelp=0.03):
            im = img.clone()
            if torch.rand(1) > p: return im
            prob = torch.rand_like(im)
            im[prob < pixelp] = im.max()
            return im

        self.aug = T.Compose([
            T.ColorJitter(hue=0.3),
            T.Lambda(lambda x: pixel_cut(x)),
            T.Lambda(lambda x: pixel_saturation(x)),
            T.Lambda(lambda x: h_linecut(x, size=2)),
            T.Lambda(lambda x: h_linecut(x, size=2)),
            T.Lambda(lambda x: h_linecut(x, size=2)),
            T.Lambda(lambda x: h_linecut(x, size=2)),
            T.Lambda(lambda x: v_linecut(x, size=2)),
            T.Lambda(lambda x: v_linecut(x, size=2)),
            T.Lambda(lambda x: v_linecut(x, size=2)),
            T.Lambda(lambda x: v_linecut(x, size=2)),
        ])

    def __call__(self, *images):
        return [self.aug(img) for img in images] if len(images) > 1 else self.aug(images[0])
