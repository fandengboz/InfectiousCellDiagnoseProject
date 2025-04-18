# -*- coding: UTF-8 -*-
"""
*@ description: 语义分割 训练
*@ name:	train.py
*@ author: dengbozfan
*@ time:	2025/04/17 10:30
"""

from .. import nn,torch,os,tqdm,ic,csv
from .. import DataLoader, Optimizer, _LRScheduler
from .. import( 
    Accumulator,SegmentationMetric,
    evaluate_segmentation_loss,evaluate_segmentation_iou
)
from .. import PlotResult
from typing import Optional
from datetime import datetime

class Train:
    """训练类, 用于管理模型的训练和测试过程"""
    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 run_dir : Optional[str] = None,
                 init_type = 'xavier',
                 device: Optional[str] = None
                 ):
        """
        初始化 Train类
        Args:
            model (nn.Module) : 实例化后的网络模型
            num_classes (int) : 分类数目
            train_loader (DataLoader): 训练集加载器
            test_loader (DataLoader): 测试集加载器
            val_loader (Optional[DataLoader]): 验证集数据加载器. 默认为None
            run_dir (Optional[str]): 训练信息存储文件目录 >> 最佳模型 | 检查点 | 训练指标
            init_type (str): 模型参数初始化类型. 默认为 'xavier'
        """
        self.model = model
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.init_type = init_type

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 检查点 
        self.best_test_iou = 0.0
        self.best_test_loss = float('inf')
        # 初始化 run_dir
        self._initialize_run_dir()
        # 初始化模型参数
        self._init_weights()

    def _initialize_run_dir(self):
        """初始化 run_dir, 确保目录存在并创建必要的子目录"""

        if self.run_dir is None:
            # 如果 run_dir 未提供，则动态生成一个默认路径
            self.run_dir = os.path.join(os.getcwd(), "runs")

        self.run_count = self._get_run_count()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_subdir = os.path.join(self.run_dir, f"run{self.run_count + 1}_{timestamp}")
        ic(self.run_subdir)
        os.makedirs(self.run_subdir, exist_ok=True)

        # 创建必要的子目录
        os.makedirs(os.path.join(self.run_subdir, "config"), exist_ok=True)
        os.makedirs(os.path.join(self.run_subdir, "model"), exist_ok=True)

        # 设置模型存储路径
        self.best_model_path = os.path.join(self.run_subdir, "model", "best.pt")
        self.checkpoint_path = os.path.join(self.run_subdir, "model", "checkpoint.pt")
        self.train_log_path = os.path.join(self.run_subdir, "model", "trainLog.csv")

    def _get_run_count(self) -> int:
        """获取当前 run_path 下已有的 run 数量"""
        
        if not os.path.exists(self.run_dir):
            return 0
        run_dirs = [d for d in os.listdir(self.run_dir) if d.startswith("run")]
        return len(run_dirs)
    
    def _load_checkpoint(self,
                         checkpoint_path:str,
                         optimizer: Optimizer,
                         scheduler: Optional[_LRScheduler] = None) -> int:
        """
        从检查点加载模型、优化器和学习率调度器的状态。

        Args:
            checkpoint_path (str): 检查点文件路径。

        Returns:
            int: 当前 epoch。
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_test_iou = checkpoint.get('best_test_iou', 0.0)
        print(f"Model loaded from {checkpoint_path}")
        print(f"Best Validation Accuracy: {self.best_test_iou:.2f}%")

        return checkpoint['epoch']
    
    def _init_weights(self):
        """初始化模型参数"""

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                if self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    raise ValueError(f"Unsupported initialization type: {self.init_type}")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    raise ValueError(f"Unsupported initialization type: {self.init_type}")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        print(f"Model parameters initialized with {self.init_type} initialization.")
    
    def _train_net_epoch(self,
                         train_loader:DataLoader,
                         optimizer: Optimizer,
                         loss_func,
                         accum_iter:int = 1,
                         ):
        """训练模型 - one epoch"""
        """
        训练单批次网络模型
        Args:
            train_loader (DataLoader): 训练集加载器
            optimizer    (Optimizer) : 模型优化器
            loss_func    (function)  : 损失函数
            accum_iter   (int)       : 累计梯度次数,每 accum_iter 个 batch 更新一次参数
        """
        
        # 初始化指标记录器
        metric_log = Accumulator(4)
        metric = SegmentationMetric(4,self.device)
        with tqdm(train_loader, unit="batch") as train_bar:
            for i,(image, mask) in enumerate(train_bar):
                image,mask = image.to(self.device), mask.to(self.device)
                # 前向传播
                output = self.model(image)
                # 计算损失
                loss = loss_func(output,mask)
                # 反向传播
                loss.backward()
                if (i + 1) % accum_iter == 0: 
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    output_class = torch.argmax(output, dim=1)
                    metric.addBatch(output_class, mask)
                    mean_iou = metric.meanIntersectionOverUnion()
                    metric_log.add(float(loss.mean()), len(image), mean_iou, 1)
                # 更新进度条显示
                train_bar.set_postfix(
                    batch_loss=float(loss.mean()),
                    avg_iou=metric_log[2]/metric_log[3]
                )
        # 计算平均损失
        train_loss = metric_log[0] / metric_log[3]
        train_iou = metric_log[2] / metric_log[3]

        return train_loss, train_iou
    
    def train(self,
              num_epochs: int,
              optimizer: Optimizer,
              loss_func: nn.Module,
              scheduler: Optional[_LRScheduler] = None,
              save_checkpoint: bool = False,
              checkpoint_interval: int = 10,
              save_best_model: bool = True,
              resume_from_checkpoint:Optional[str] = None
              ):
        """
        训练模型
        Args:
            num_epochs(int): 训练的轮数
            optimizer (torch.optim.Optimizer): 优化器
            loss_func (nn.Module): 损失函数
            scheduler (Optional[_LRScheduler]): 学习率调度器。默认为 None。
            save_checkpoint (bool): 是否保存训练检查点。默认为 False。
            checkpoint_interval (int): 保存检查点的间隔(单位:epoch)。默认为 10。
            save_best_model (bool): 是否保存最佳模型。默认为 True。 
            resume_from_checkpoint (Optional[str]): 从指定的检查点文件恢复训练。默认为 None。 
          
        """
        start_epoch = 0

        # 如果提供了检查点路径，则从检查点恢复训练
        if resume_from_checkpoint is not None:
            start_epoch = self._load_checkpoint(resume_from_checkpoint, optimizer, scheduler)
            print(f"Resuming training from epoch {start_epoch + 1}")
        
        print(f"开始在{self.device} 上进行训练")
        self.model.to(self.device)

        for epoch in range(start_epoch, num_epochs):
            # 计算一次训练后的loss和iou
            train_loss, train_iou = self._train_net_epoch(self.train_loader,optimizer,loss_func,accum_iter=1)
            # 计算当前的 test/val loss 和 test iou
            self.model.eval()
            # 计算验证集的损失
            if self.val_loader is not None:
                val_loss = evaluate_segmentation_loss(self.model, self.val_loader, loss_func, self.device)
            else:
                val_loss = evaluate_segmentation_loss(self.model, self.test_loader, loss_func, self.device)
            # 计算测试集的IoU
            test_iou = evaluate_segmentation_iou(self.model, self.test_loader, self.num_classes, self.device)
            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step(val_loss)
            # 存储训练数据 | 学习曲线
            self._save_train_learn_data(epoch,train_loss,train_iou,val_loss,test_iou,current_lr)
            # 保存最佳模型
            if save_best_model:
                self._save_best_model(self.best_model_path,test_iou)
            # 保存检查点
            if save_checkpoint and (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(self.checkpoint_path,epoch+1,optimizer,scheduler)
            # 每 2 个epoch绘制一次训练曲线
            if (epoch + 1) % 2 == 0:
                self.save_train_data(epoch)

        # 最后保存训练数据
        self.save_train_data(epoch)

        print("训练完成了！😊🎉ヾ(≧▽≦*)o")

    def save_train_data(self,epoch):
        """保存训练结果"""

        self.trainResult_save_dir = f'{self.run_subdir}\config\config_{epoch+1}'

        os.makedirs(self.trainResult_save_dir,exist_ok=True)
        plot = PlotResult(train_log_path=self.train_log_path,save_dir=self.trainResult_save_dir)

        plot.plotAll()
        plot.compute_and_plot_confusion_matrix(epoch,self.num_classes,self.model,self.test_loader,self.device)

    def _save_train_learn_data(self,epoch:int, train_loss:float, train_iou:float,val_loss:float,test_iou:float,current_lr:float):
        """
        存储 训练学习数据 | 绘制学习曲线
        Args:
            epoch       (int)    : 当前训练批次
            train_loss  (float)  : 训练损失
            train_iou   (float)  : 训练Iou
            val_loss    (float)  : 验证损失
            test_iou    (float)  : 测试iou
            current_lr  (float)  : 当前学习率
        """
        with open(self.train_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if epoch == 0:
                writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Test Acc', 'LR'])
            
            writer.writerow([epoch, f"{train_loss:.5f}", f"{train_iou:.5f}", f"{val_loss:.5f}", f"{test_iou:.3f}",f'{current_lr:.6f}'])
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.5f}, Train Acc: {train_iou:.5f}, Val Loss: {val_loss:.5f}, Test Acc: {test_iou:.5f}, LR: {current_lr:.5f}")
            

    def _save_best_model(self,path:str, test_iou: float):
        """ 
        保存最佳模型
        Args:
            path (str): 最佳模型存储路径
            test_iou (float) : 当前训练后计算得出的iou
        """
        if test_iou > self.best_test_iou:
            self.best_test_iou = test_iou
            torch.save(self.model.state_dict(), path)
            print(f'已保存当前最佳模型, Best Test IoU: {self.best_test_iou:.4f}')
        

    def _save_checkpoint(self, path: str, epoch: int, optimizer: Optimizer, scheduler: Optional[_LRScheduler]):
        """
        保存训练检查点。

        Args:
            path (str): 检查点保存路径。
            epoch (int): 当前 epoch。
            optimizer (Optimizer): 优化器。
            scheduler (Optional[_LRScheduler]): 学习率调度器。
        """
        if self.run_dir is not None:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_test_loss': self.best_test_loss,
                'best_test_iou': self.best_test_iou
            }
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")
