import torch


def Loss(Prediction, target, loss_obj):

    # 一个认定语句中Padding部分的Mask（Padding部分的id是0）
    mask = 1 - torch.eq(target, 0).long()  # 现在mask中Padding的位置是0，非Padding部分是1

    # 计算损失
    loss = loss_obj(Prediction, target.long())

    # 将无效的损失值删除
    mask = mask.to(dtype=loss.dtype)
    loss *= mask  # Padding部分的损失全为 0

    # 返回平均值
    return torch.mean(loss)
