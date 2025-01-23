import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import numpy as np
from convert_tensorrt import convert_tensorrt
# from eval_with_detector import eval_w_detector
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from io import BytesIO
import PIL
from torchvision.transforms import ToTensor
import json
import matplotlib.pyplot as plt


def draw_precision_curve(precision_baseline, precision_processed):


    x = np.arange(0, 1.01, 0.01)
    plt.title("Precision-Recall Eğrisi")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(x, precision_baseline, label='P_baseline')
    plt.plot(x, precision_processed, label='P_processed')

    plt.legend(["Baseline Precision", "Processed Precision"])
    plt.grid()
    buf = BytesIO()
    plt.savefig(buf, format='png')


    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()

    return image


class Classifier(pl.LightningModule):

    def __init__(self, model, scheduler, optimizer,
                 loss=None, save_every_epoch=True, 
                 ckpt_path=None, num_classes=2, img_size=128, model_name='tf_efficientnet_lite0'):
        super().__init__()

        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.best_precision = 0.0
        self.best_model_pth = ''
        self.save_every_epoch = save_every_epoch
        self.thresholds = np.linspace(0,1,21)
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_name = model_name
        self.sanity_check_done = False
        self.softmax = nn.Softmax(dim=1)
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss

        if ckpt_path is not None:
            self.ckpt_path = os.path.join(ckpt_path, 'weights')
        else:
            self.ckpt_path = 'weights'

        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path)

    def forward(self, x):

        return self.model(x)


    def training_step(self, train_batch, batch_idx):

        img, label, _ = train_batch
        
        output = self.forward(img)
        
        _, preds = torch.max(output, 1)
        
        loss = self.loss(output, label)
        
        corrects = torch.sum(preds == label.data) / img.shape[0]
        
        if self.scheduler.__class__ == torch.optim.lr_scheduler.CosineAnnealingLR:
            self.scheduler.step()
            
        self.log('learning rate', self.scheduler.get_lr()[0])
        
        return {'loss': loss, 'corrects': corrects}


    def training_epoch_end(self, outputs):

        losses = [x['loss'] for x in outputs]
        corrects = [x['corrects'] for x in outputs]
        
        avg_train_loss = sum(losses) / len(losses)
        train_epoch_accuracy = sum(corrects) / len(corrects)

        self.log('train accuracy', train_epoch_accuracy)
        self.log('train loss', avg_train_loss)


    def validation_step(self, val_batch, batch_idx):
        img, label, _ = val_batch
        output = self.forward(img)
        _, preds = torch.max(output, 1)
        scores = self.softmax(output).cpu().numpy()

        loss = self.loss(output, label)

        corrects = torch.sum(preds == label.data) / img.shape[0]

        return {'loss': loss, 'corrects': corrects, 'scores': scores, 'labels': label.cpu().numpy()}



    def validation_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        corrects = [x['corrects'] for x in outputs]

        APs = []
        all_labels = np.concatenate([x['labels'] for x in outputs])

        for i in range(self.num_classes):
            all_scores = np.concatenate([x['scores'][:, i] for x in outputs])
            precision, recall, thresholds = precision_recall_curve(all_labels, all_scores, pos_label=i)
            average_precision = average_precision_score(all_labels, all_scores, pos_label=i)
            APs.append(average_precision)
            self.log(f'class {i} average precision', average_precision)

            PR_curve = { 
                        i: [np.float64(x), np.float64(y), np.float64(z)] for i, x, y, z in zip(range(1, len(precision)), precision[:-1], recall[:-1], thresholds)
                    
                    }
            
            PR_curve_pth = f'{self.ckpt_path}'.replace('weights', f'PR_curves_class_{i}')

            if not os.path.isdir(PR_curve_pth):
                os.makedirs(PR_curve_pth)

            PrecisionRecallDisplay.from_predictions(all_labels, all_scores, pos_label=i)
            plt.ylim(ymin=0)
            plt.savefig(f'{PR_curve_pth}/PR_curve_epoch_{self.current_epoch}')

            with open(f'{PR_curve_pth}/PR_curve_class_{i}_epoch_{self.current_epoch}.json', 'w') as f:
                json.dump(PR_curve, f)

        average_precision = APs[0]
        # mAP = sum(APs) / len(APs)
        # average_precision = mAP
        # all_scores = np.concatenate([x['scores'][:, 0] for x in outputs])
        # all_labels = np.concatenate([x['labels'] for x in outputs])
        # precision, recall, thresholds = precision_recall_curve(all_labels, all_scores, pos_label=0)
        # average_precision = average_precision_score(all_labels, all_scores, pos_label=0)
        # f1_scores = 2 * recall * precision / (recall + precision)
        #best_thresh = thresholds[np.argmax(f1_scores)]

        avg_val_loss = sum(losses) / len(losses)
        val_epoch_accuracy = sum(corrects) / len(corrects)

        if average_precision >= self.best_precision:
            self.best_precision = average_precision
            self.best_model_pth = f'{self.ckpt_path}/best_epoch_{self.current_epoch}_precision_{int(self.best_precision * 100)}.ckpt'
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
                }, self.best_model_pth)
            

            



        if self.save_every_epoch:
            torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, f'{self.ckpt_path}/epoch_{self.current_epoch}_precision_{int(average_precision* 100)}.ckpt')

        if not self.sanity_check_done:
            # Assuming this is the end of the sanity check
            self.best_precision = 0.0
            self.best_model_pth = ''
            self.sanity_check_done = True

        self.log('val loss', avg_val_loss)
        self.log('val accuracy', val_epoch_accuracy)
        self.log('average precision', average_precision)
                
        self.log('best average precision', self.best_precision)

    # def test_tensorrt(self):

    #     model_trt, model_pth = convert_tensorrt(self.best_model_pth, num_classes=self.num_classes, input_size=self.img_size, save=True, model_name=self.model_name)

    #     stats_baseline, pr_curve_baseline, precision_baseline = eval_w_detector(ckpt_path=model_pth, pred_pth=self.pred_detections_pth, 
    #                     bbox_crop_path=self.bbox_infer_dataset, out_path='out.json', 
    #                     input_size=self.img_size, ann_path=self.annotations_pth, baseline=True)
                
    #     stats_processed, pr_curve_processed, precision_processed = eval_w_detector(ckpt_path=model_pth, pred_pth=self.pred_detections_pth, 
    #                     bbox_crop_path=self.bbox_infer_dataset, out_path='out.json', 
    #                     input_size=self.img_size, ann_path=self.annotations_pth)


    #     pr_curve_combined = draw_precision_curve(precision_processed=precision_processed,
    #                          precision_baseline=precision_baseline)     

    #     self.logger.experiment.add_image('PR curve Baseline', pr_curve_baseline, self.current_epoch)
    #     self.logger.experiment.add_image('PR curve Processed', pr_curve_processed, self.current_epoch)
    #     self.logger.experiment.add_image('PR curve Combined', pr_curve_combined, self.current_epoch)


    #     for stats, name in ([stats_baseline, 'Baseline'], [stats_processed, 'Processed']):
    #         self.logger.experiment.add_scalar(f'{name} (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', stats[0])
    #         self.logger.experiment.add_scalar(f'{name} (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]', stats[1])
    #         self.logger.experiment.add_scalar(f'{name} (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]', stats[2])
    #         self.logger.experiment.add_scalar(f'{name} (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', stats[3])
    #         self.logger.experiment.add_scalar(f'{name} (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', stats[4])
    #         self.logger.experiment.add_scalar(f'{name} (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', stats[5])
    #         self.logger.experiment.add_scalar(f'{name} (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]', stats[6])
    #         self.logger.experiment.add_scalar(f'{name} (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]', stats[7])

    #         self.logger.experiment.add_scalar(f'{name} (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', stats[8])
    #         self.logger.experiment.add_scalar(f'{name} (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', stats[9])
    #         self.logger.experiment.add_scalar(f'{name} (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', stats[10])
    #         self.logger.experiment.add_scalar(f'{name} (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', stats[11])

    #     return stats


    def configure_optimizers(self):

        optimizer = self.optimizer
        scheduler = self.scheduler
        
        if scheduler:
            return [optimizer], [scheduler]
        
        return [optimizer]

