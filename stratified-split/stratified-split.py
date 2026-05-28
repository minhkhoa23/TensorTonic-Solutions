import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    if rng is None:
        # Nếu không có rng, dùng cơ chế shuffle mặc định của numpy
        shuffle_func = np.random.shuffle
    elif isinstance(rng, (int, np.integer)):
        _rng = np.random.default_rng(rng)
        shuffle_func = _rng.shuffle
    else:
        shuffle_func = rng.shuffle

    X = np.asarray(X)
    y = np.asarray(y)
    
    train_indices = []
    test_indices = []
    
    # Hint 1: Lấy các class duy nhất
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        # Hint 2: Tìm indices của từng class
        cls_indices = np.where(y == cls)[0]
        
        # Shuffle các index trong nội bộ class đó
        shuffle_func(cls_indices)
        
        # Tính toán số lượng mẫu cho tập test dựa trên test_size
        n_samples = len(cls_indices)
        n_test = int(np.round(test_size * n_samples))
        
        # Đảm bảo ràng buộc: Luôn giữ lại ít nhất 1 mẫu cho tập train nếu có thể
        if n_samples >= 2 and (n_samples - n_test) < 1:
            n_test = n_samples - 1
            
        # Chia tách index (Lấy phần đuôi làm test, phần đầu làm train)
        cls_test = cls_indices[:n_test]
        cls_train = cls_indices[n_test:]
        
        test_indices.extend(cls_test)
        train_indices.extend(cls_train)
        
    # QUAN TRỌNG: Sắp xếp lại index tăng dần để bảo toàn thứ tự gốc của mảng
    train_indices = np.sort(np.array(train_indices))
    test_indices = np.sort(np.array(test_indices))
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]