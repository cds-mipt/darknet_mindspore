import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="GPU", distribute=False, num_parallel_workers=6):
    """
    create a train or eval imagenet2012 dataset for darknet53

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        num_parallel_workers(int)

    Returns:
        dataset(mindspore.dataset)
    """

    dataset = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=True)
    
    image_size = 256
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(292),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)
    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=num_parallel_workers)
    dataset = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)

    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    dataset = dataset.repeat(repeat_num)

    return dataset