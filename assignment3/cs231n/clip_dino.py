from tensorflow.python.framework.ops import device_v2
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import tensorflow_datasets as tfds
from torchvision import transforms as T
import cv2
from tqdm.auto import tqdm


def get_similarity_no_loop(text_features, image_features):
    """
    Computes the pairwise cosine similarity between text and image feature vectors.

    Args:
        text_features (torch.Tensor): A tensor of shape (N, D).
        image_features (torch.Tensor): A tensor of shape (M, D).

    Returns:
        torch.Tensor: A similarity matrix of shape (N, M), where each entry (i, j)
        is the cosine similarity between text_features[i] and image_features[j].
    """
    similarity = None
    ############################################################################
    # TODO: Compute the cosine similarity. Do NOT use for loops.               #
    ############################################################################

    # `text_features[:, None]` 会在第 1 维插入一个长度为 1 的新维度。
    # 如果原来 `text_features` 的形状是 `(N, D)`，现在就会变成 `(N, 1, D)`。
    # 这样它就能和形状为 `(M, D)` 的 `image_features` 自动广播成 `(N, M, D)`。
    # `dim=-1` 表示在最后一个维度 `D` 上计算余弦相似度，也就是按特征维做比较。
    similarity = nn.functional.cosine_similarity(
        text_features[:, None], image_features, dim=-1
    )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """Performs zero-shot image classification using a CLIP model.

    Args:
        clip_model (torch.nn.Module): The pre-trained CLIP model for encoding
            images and text.
        clip_preprocess (Callable): A preprocessing function to apply to each
            image before encoding.
        images (List[np.ndarray]): A list of input images as NumPy arrays
            (H x W x C) uint8.
        class_texts (List[str]): A list of class label strings for zero-shot
            classification.
        device (torch.device): The device on which computation should be
            performed. Pass text_tokens to this device before passing it to
            clip_model.

    Returns:
        List[str]: Predicted class label for each image, selected from the
            given class_texts.
    """
    
    pred_classes = []

    ############################################################################
    # TODO: Find the class labels for images.                                  #
    ############################################################################

    # 先把所有类别文本一次性转成 CLIP 需要的 token 编号。
    # `clip.tokenize(class_texts)` 的输出形状大致是 `(类别数, token长度)`。
    # 再用 `.to(device)` 把这个张量放到 CPU 或 GPU 上，保证后面能和模型在同一设备运行。
    text_tokens = clip.tokenize(class_texts).to(device)

    # 把每个类别文本编码成一个特征向量。
    # 这些文本特征和图像特征会落到同一个共享语义空间里，后面才能直接比较谁更相似。
    text_features = clip_model.encode_text(text_tokens)

    # 这里用列表推导式逐张处理图片：
    # 1. `Image.fromarray(img)` 把 numpy 图片转成 PIL 图片；
    # 2. `clip_preprocess(...)` 按 CLIP 的要求做 resize、裁剪、归一化；
    # 3. 最终得到每张图片对应的张量。
    processed_images = [clip_preprocess(Image.fromarray(img)) for img in images]

    # `torch.stack(processed_images)` 会把“很多张单独的图片张量”拼成一个批量张量。
    # 这样模型就可以一次前向传播编码整批图片，而不是一张一张单独跑。
    images_tensor = torch.stack(processed_images).to(device)
    image_features = clip_model.encode_image(images_tensor)

    # `sims` 是文本和图片两两之间的相似度矩阵，形状是 `(类别数, 图片数)`。
    # `torch.argmax(sims, axis=0)` 表示“按列取最大值的下标”：
    # 对每一张图片，都找出和它最相似的那个类别编号。
    # 最后再用列表推导式把“类别编号”换回对应的“类别文本”。
    sims = get_similarity_no_loop(text_features, image_features)
    pred_classes = [class_texts[i] for i in torch.argmax(sims, axis=0)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    A simple image retrieval system using CLIP.
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        Args:
          clip_model (torch.nn.Module): The pre-trained CLIP model.
          clip_preprocess (Callable): Function to preprocess images.
          images (List[np.ndarray]): List of images as NumPy arrays (H x W x C).
          device (torch.device): The device for model execution.
        """
        ############################################################################
        # TODO: Store all necessary object variables to use in retrieve method.    #
        # Note that you should process all images at once here and avoid repeated  #
        # computation for each text query. You may end up NOT using the above      #
        # similarity function for most compute-optimal implementation.#
        ############################################################################
        
        # 先把模型和设备保存成对象属性。
        # 这样在 `retrieve()` 里就可以直接复用，不需要每次查询都重新传进来。
        self.clip_model = clip_model
        self.device = device

        # 这里在初始化阶段就把所有图片先编码成特征。
        # 好处是：后面每次输入新的文本查询时，不用重复计算图片特征，只需要算一次文本特征即可。
        # 这是一种典型的“把固定不变的结果提前缓存起来”的优化方式。
        processed_images = [clip_preprocess(Image.fromarray(img)) for img in images]
        images_tensor = torch.stack(processed_images).to(device)
        self.image_features = clip_model.encode_image(images_tensor)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        Retrieves the indices of the top-k images most similar to the input text.
        You may find torch.Tensor.topk method useful.

        Args:
            query (str): The text query.
            k (int): Return top k images.

        Returns:
            List[int]: Indices of the top-k most similar images.
        """
        top_indices = []
        ############################################################################
        # TODO: Retrieve the indices of top-k images.                              #
        ############################################################################

        # 即使这里只有一个查询字符串，也要写成 `[query]`。
        # 因为 `clip.tokenize` 默认把输入当成“一个文本批次”来处理，而不是单个字符串。
        text_tokens = clip.tokenize([query]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)

        # 这里得到的 `sims` 形状是 `(1, 图片数)`，因为只有 1 条查询文本。
        # 所以 `sims[0]` 就是这条查询和所有图片之间的相似度。
        # `torch.argsort(..., descending=True)` 会把下标按“从大到小”排序，
        # 也就是把最相似的图片排在最前面。
        # `tolist()[:k]` 再取前 `k` 个结果，作为检索到的图片编号。
        sims = get_similarity_no_loop(text_features, self.image_features)
        top_indices = torch.argsort(sims[0], descending=True).tolist()[:k]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return top_indices

  
class DavisDataset:
    def __init__(self):
        self.davis = tfds.load('davis/480p', split='validation', as_supervised=False)
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        
      
    def get_sample(self, index):
        assert index < len(self.davis)
        ds_iter = iter(tfds.as_numpy(self.davis))
        for i in range(index+1):
            video = next(ds_iter)
        frames, masks = video['video']['frames'], video['video']['segmentations']
        print(f"video {video['metadata']['video_name'].decode()}  {len(frames)} frames")
        return frames, masks
    
    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
              tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])

        res = torch.stack(res)
        return res
    
    def process_masks(self, masks, device):
        res = []
        for m in masks:
            m = cv2.resize(m, (60,60), cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)
        return res
    
    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()
        mask = mask.reshape((60, 60))
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay
        


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    Generate a colored segmentation overlay on top of an RGB image.

    Parameters:
        segmentation_mask (np.ndarray): 2D array of shape (H, W), with class indices.
        image (np.ndarray): 3D array of shape (H, W, 3), RGB image.
        alpha (float): Transparency factor for overlay (0 = only image, 1 = only mask).

    Returns:
        np.ndarray: Image with segmentation overlay (shape: (H, W, 3), dtype: uint8).
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # Generate deterministic colors for each class using a fixed colormap
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # Create a color image for the segmentation mask
    seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """Compute the mean Intersection over Union (IoU)."""
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        Initialize the DINOSegmentation model.

        This defines a simple neural network designed to  classify DINO feature
        vectors into segmentation classes. It includes model initialization,
        optimizer, and loss function setup.

        Args:
            device (torch.device): Device to run the model on (CPU or CUDA).
            num_classes (int): Number of segmentation classes.
            inp_dim (int, optional): Dimensionality of the input DINO features.
        """

        ############################################################################
        # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
        # function to train classify each DINO feature vector into a seg. class.   #
        # It can be a linear layer or two layer neural network.                    #
        ############################################################################

        # `nn.Sequential(...)` 表示把多层网络按顺序串起来执行。
        # 这一小段网络的流程是：
        # 1. `Linear` 把输入特征从 `inp_dim` 投影到更小的隐藏维度；
        # 2. `BatchNorm1d` 做批归一化，让训练更稳定；
        # 3. `GELU()` 加入非线性；
        # 4. 最后一个 `Linear` 输出每个类别对应的分数（logits）。
        self.nn = nn.Sequential(
            nn.Linear(inp_dim, inp_dim // 2),
            nn.BatchNorm1d(inp_dim // 2),
            nn.GELU(),
            nn.Linear(inp_dim // 2, num_classes),
        ).to(device)

        # `AdamW` 是优化器，负责根据梯度更新网络参数。
        # `self.nn.parameters()` 表示把这个小网络里所有可训练参数都交给优化器管理。
        # `weight_decay=0.1` 是权重衰减，相当于一种简单正则化，可以减轻过拟合。
        self.optim = torch.optim.AdamW(self.nn.parameters(), weight_decay=0.1)

        # `CrossEntropyLoss` 用来做多分类。
        # 它要求输入是原始分数 `logits`，标签是整数类别编号。
        # 注意这里不需要自己手动写 softmax，因为这个损失函数内部已经包含了相应计算。
        self.loss_fn = nn.CrossEntropyLoss()

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass

    def train(self, X_train, Y_train, num_iters=500):
        """Train the segmentation model using the provided training data.

        Args:
            X_train (torch.Tensor): Input feature vectors of shape (N, D).
            Y_train (torch.Tensor): Ground truth labels of shape (N,).
            num_iters (int, optional): Number of optimization steps.
        """
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################

        for _ in (pbar := tqdm(range(num_iters), desc="Training")):
            # `zero_grad()` 先把上一次迭代留下的梯度清空。
            # 这是因为 PyTorch 默认会把多次 `backward()` 的梯度累加起来，
            # 如果不清空，就会把前几轮的梯度也一起算进去。
            self.optim.zero_grad()

            # 前向传播：把输入的 DINO 特征送进小网络，得到每个样本属于各类别的分数。
            # `X_pred` 的每一行都对应一个样本，每一列对应一个类别的 logits。
            X_pred = self.nn(X_train)
            loss = self.loss_fn(X_pred, Y_train)

            # `loss.backward()` 会根据当前损失，自动计算网络中每个参数的梯度。
            # `self.optim.step()` 会真正根据这些梯度更新参数，完成一次训练步。
            loss.backward()
            self.optim.step()

            # `loss.item()` 把只有一个数的张量取成普通 Python 数值。
            # `set_postfix(...)` 会把这个损失值显示在进度条后面，便于观察训练是否在下降。
            pbar.set_postfix(loss=loss.item())

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def inference(self, X_test):
        """Perform inference on the given test DINO feature vectors.

        Args:
            X_test (torch.Tensor): Input feature vectors of shape (N, D).

        Returns:
            torch.Tensor of shape (N,): Predicted class indices.
        """
        pred_classes = None
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################

        # 推理时不需要再算损失，只需要取每一行里分数最大的那个类别。
        # `dim=1` 表示沿着“类别这一维”找最大值。
        # 最终得到的 `pred_classes` 是一个一维张量，里面存的是每个样本的预测类别编号。
        pred_classes = torch.argmax(self.nn(X_test), dim=1)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return pred_classes