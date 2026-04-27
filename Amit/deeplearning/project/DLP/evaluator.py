import torch

class Evaluator:
    def evaluate(self, model, dataloader):
        # 1. بنقفل وضع التدريب ونشغل وضع الامتحان
        model.eval() 
        
        correct = 0
        total = 0
        
        # 2. بنوقف حسابات التعديل (عشان ده امتحان مش وقت تعليم)
        with torch.no_grad(): 
            for images, labels in dataloader:
                # بنفرد الصور زي ما عملنا في التدريب
                images = images.view(images.shape[0], -1) 
                
                # الموديل بيطلع توقعاته
                outputs = model(images) 
                
                # بنختار أعلى رقم (أعلى احتمالية الموديل اختارها)
                _, predicted = torch.max(outputs.data, 1) 
                
                # بنجمع عدد الصور الكلي
                total += labels.size(0) 
                
                # بنزود العداد لو إجابة الموديل طابقت الإجابة الصح
                correct += (predicted == labels).sum().item() 
        
        # 3. بنحسب النسبة المئوية
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy