"""
Evaluation metrics for document extraction tasks
"""
import json
from typing import Dict, List, Any
from collections import defaultdict


def evaluate_docred(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估 DocRED 关系抽取
    
    Metrics: Precision, Recall, F1 for relation triplets
    """
    pred_relations = []
    gold_relations = []
    
    for pred, gold in zip(predictions, golds):
        # 解析预测结果
        try:
            pred_data = json.loads(pred.get("prediction", "{}"))
            pred_rels = pred_data.get("relations", [])
            pred_relations.extend([
                (r.get("head", ""), r.get("relation", ""), r.get("tail", ""))
                for r in pred_rels
            ])
        except:
            pass
        
        # 解析金标准
        try:
            gold_data = json.loads(gold) if isinstance(gold, str) else gold
            gold_rels = gold_data if isinstance(gold_data, list) else gold_data.get("relations", [])
            gold_relations.extend([
                (r.get("head", ""), r.get("relation", ""), r.get("tail", ""))
                for r in gold_rels
            ])
        except:
            pass
    
    pred_set = set(pred_relations)
    gold_set = set(gold_relations)
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "pred_count": len(pred_set),
        "gold_count": len(gold_set)
    }


def evaluate_cord(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估 CORD 收据抽取（官方格式）
    
    Metrics: Field-level accuracy for num_items, subtotal_price, service_price, tax_price, total_price, etc
    """
    correct_fields = defaultdict(int)
    total_fields = defaultdict(int)
    
    # Official CORD fields
    eval_fields = ["num_items", "subtotal_price", "service_price", "tax_price", "total_price", "etc"]
    
    for pred, gold in zip(predictions, golds):
        try:
            pred_data = json.loads(pred.get("prediction", "{}"))
            gold_data = json.loads(gold) if isinstance(gold, str) else gold
        except:
            continue
        
        # 评估每个字段
        for field in eval_fields:
            pred_val = str(pred_data.get(field, "")).strip()
            gold_val = str(gold_data.get(field, "")).strip()
            
            if pred_val and gold_val:  # Both have value
                if pred_val == gold_val:
                    correct_fields[field] += 1
                total_fields[field] += 1
            elif not gold_val:  # Gold doesn't have this field, skip
                continue
            else:  # Pred missing but gold has value
                total_fields[field] += 1
    
    # 计算准确率
    results = {}
    for field in eval_fields:
        if total_fields[field] > 0:
            accuracy = correct_fields[field] / total_fields[field]
            results[f"{field}_accuracy"] = accuracy
    
    # 总体准确率
    total_correct = sum(correct_fields.values())
    total_count = sum(total_fields.values())
    results["overall_accuracy"] = total_correct / total_count if total_count > 0 else 0.0
    
    return results


def evaluate_funsd(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估 FUNSD 表单理解
    
    Metrics: Entity-level and Relation-level F1
    """
    # Entity evaluation
    pred_entities = []
    gold_entities = []
    
    # Relation evaluation
    pred_relations = []
    gold_relations = []
    
    for pred, gold in zip(predictions, golds):
        try:
            # Handle prediction - may be string or dict
            if isinstance(pred, dict):
                pred_str = pred.get("prediction", "{}")
            else:
                pred_str = str(pred) if pred else "{}"
            pred_data = json.loads(pred_str) if isinstance(pred_str, str) else pred_str
            
            # Handle gold - may be string or dict
            gold_data = json.loads(gold) if isinstance(gold, str) else gold
            
            # Ensure both are dicts
            if not isinstance(pred_data, dict):
                pred_data = {}
            if not isinstance(gold_data, dict):
                gold_data = {}
        except Exception as e:
            print(f"[Warning] Failed to parse prediction/gold: {e}")
            pred_data = {}
            gold_data = {}
            continue
        
        # Extract entities
        pred_ents = pred_data.get("entities", [])
        gold_ents = gold_data.get("entities", [])
        
        pred_entities.extend([(e.get("text", ""), e.get("label", "")) for e in pred_ents])
        gold_entities.extend([(e.get("text", ""), e.get("label", "")) for e in gold_ents])
        
        # Extract relations
        pred_rels = pred_data.get("relations", [])
        gold_rels = gold_data.get("relations", [])
        
        pred_relations.extend([(r.get("head", ""), r.get("tail", ""), r.get("type", "")) for r in pred_rels])
        gold_relations.extend([(r.get("head", ""), r.get("tail", ""), r.get("type", "")) for r in gold_rels])
    
    # Calculate entity metrics
    pred_ent_set = set(pred_entities)
    gold_ent_set = set(gold_entities)
    
    ent_tp = len(pred_ent_set & gold_ent_set)
    ent_fp = len(pred_ent_set - gold_ent_set)
    ent_fn = len(gold_ent_set - pred_ent_set)
    
    ent_precision = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) > 0 else 0.0
    ent_recall = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) > 0 else 0.0
    ent_f1 = 2 * ent_precision * ent_recall / (ent_precision + ent_recall) if (ent_precision + ent_recall) > 0 else 0.0
    
    # Calculate relation metrics
    pred_rel_set = set(pred_relations)
    gold_rel_set = set(gold_relations)
    
    rel_tp = len(pred_rel_set & gold_rel_set)
    rel_fp = len(pred_rel_set - gold_rel_set)
    rel_fn = len(gold_rel_set - pred_rel_set)
    
    rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0.0
    rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0.0
    rel_f1 = 2 * rel_precision * rel_recall / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0
    
    return {
        "entity_precision": ent_precision,
        "entity_recall": ent_recall,
        "entity_f1": ent_f1,
        "relation_precision": rel_precision,
        "relation_recall": rel_recall,
        "relation_f1": rel_f1,
        "overall_f1": (ent_f1 + rel_f1) / 2
    }


def evaluate_finer(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估 FinER-139 金融实体识别
    
    Metrics: Entity-level Precision, Recall, F1
    """
    pred_entities = []
    gold_entities = []
    
    for pred, gold in zip(predictions, golds):
        try:
            pred_data = json.loads(pred.get("prediction", "{}"))
            gold_data = json.loads(gold) if isinstance(gold, str) else gold
        except:
            continue
        
        pred_ents = pred_data.get("entities", [])
        gold_ents = gold_data.get("entities", [])
        
        # 使用 (text, label) 作为唯一标识
        pred_entities.extend([(e.get("text", ""), e.get("label", "")) for e in pred_ents])
        gold_entities.extend([(e.get("text", ""), e.get("label", "")) for e in gold_ents])
    
    pred_set = set(pred_entities)
    gold_set = set(gold_entities)
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }


def evaluate_extraction_task(task: str, predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    统一评估接口
    
    Args:
        task: 'docred', 'cord', 'funsd', 'finer'
        predictions: 预测结果列表
        golds: 金标准列表
    
    Returns:
        评估指标字典
    """
    if task == "docred":
        return evaluate_docred(predictions, golds)
    elif task == "cord":
        return evaluate_cord(predictions, golds)
    elif task == "funsd":
        return evaluate_funsd(predictions, golds)
    elif task == "finer":
        return evaluate_finer(predictions, golds)
    else:
        return {"error": f"Unknown task: {task}"}


def print_evaluation_results(task: str, metrics: Dict[str, float]):
    """
    打印评估结果
    """
    print("\n" + "="*60)
    print(f"📊 Evaluation Results for {task.upper()}")
    print("="*60)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric:40s}: {value:6.2%}")
        else:
            print(f"  {metric:40s}: {value}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # 测试示例
    test_predictions = [
        {"prediction": '{"relations": [{"head": "Apple", "relation": "founded_by", "tail": "Steve Jobs"}]}'}
    ]
    test_golds = [
        [{"head": "Apple", "relation": "founded_by", "tail": "Steve Jobs"}]
    ]
    
    metrics = evaluate_docred(test_predictions, test_golds)
    print_evaluation_results("docred", metrics)
