"""
数据可视化及信息检索与问答系统 V1.0
作者：石浩田
功能：完整的知识图谱系统，包含数据可视化、信息检索和智能问答三个核心模块
"""

from flask import Flask, render_template, jsonify, request
import json
import random
import time
from datetime import datetime
import os

app = Flask(__name__)

# ========== 模拟数据 ==========

# 知识图谱数据
KNOWLEDGE_GRAPH = {
    "nodes": [
        {"id": "AI", "label": "人工智能", "type": "概念", "group": 1, "value": 30},
        {"id": "ML", "label": "机器学习", "type": "技术", "group": 2, "value": 25},
        {"id": "DL", "label": "深度学习", "type": "技术", "group": 2, "value": 20},
        {"id": "NN", "label": "神经网络", "type": "技术", "group": 2, "value": 18},
        {"id": "NLP", "label": "自然语言处理", "type": "领域", "group": 3, "value": 22},
        {"id": "CV", "label": "计算机视觉", "type": "领域", "group": 3, "value": 22},
        {"id": "KG", "label": "知识图谱", "type": "技术", "group": 2, "value": 20},
        {"id": "LLM", "label": "大语言模型", "type": "技术", "group": 2, "value": 28},
        {"id": "DM", "label": "数据挖掘", "type": "技术", "group": 2, "value": 18},
        {"id": "BD", "label": "大数据", "type": "概念", "group": 1, "value": 25},
        {"id": "RL", "label": "强化学习", "type": "技术", "group": 2, "value": 16},
        {"id": "CV-App", "label": "人脸识别", "type": "应用", "group": 4, "value": 15},
        {"id": "NLP-App", "label": "机器翻译", "type": "应用", "group": 4, "value": 15},
        {"id": "ML-App", "label": "推荐系统", "type": "应用", "group": 4, "value": 15}
    ],
    "edges": [
        {"source": "AI", "target": "ML", "label": "包含", "value": 5},
        {"source": "AI", "target": "NLP", "label": "包含", "value": 5},
        {"source": "AI", "target": "CV", "label": "包含", "value": 5},
        {"source": "ML", "target": "DL", "label": "子类", "value": 4},
        {"source": "ML", "target": "RL", "label": "子类", "value": 4},
        {"source": "DL", "target": "NN", "label": "基于", "value": 3},
        {"source": "DL", "target": "NLP", "label": "应用于", "value": 3},
        {"source": "DL", "target": "CV", "label": "应用于", "value": 3},
        {"source": "KG", "target": "AI", "label": "支撑", "value": 4},
        {"source": "LLM", "target": "NLP", "label": "增强", "value": 4},
        {"source": "LLM", "target": "KG", "label": "补充", "value": 3},
        {"source": "BD", "target": "ML", "label": "驱动", "value": 4},
        {"source": "CV", "target": "CV-App", "label": "应用于", "value": 2},
        {"source": "NLP", "target": "NLP-App", "label": "应用于", "value": 2},
        {"source": "ML", "target": "ML-App", "label": "应用于", "value": 2},
        {"source": "DM", "target": "ML", "label": "支撑", "value": 3},
        {"source": "DM", "target": "BD", "label": "处理", "value": 3}
    ]
}

# 检索文档数据
SEARCH_DOCUMENTS = [
    {
        "id": 1,
        "title": "人工智能发展白皮书2023",
        "content": "人工智能是计算机科学的一个分支，致力于创建能够执行需要人类智能的任务的智能代理。这些任务包括学习、推理、问题解决、感知和语言理解。",
        "type": "白皮书",
        "category": "人工智能",
        "date": "2023-05-15",
        "relevance": 0.95,
        "views": 1250
    },
    {
        "id": 2,
        "title": "机器学习算法原理与应用",
        "content": "机器学习是人工智能的一个子领域，使计算机能够在没有明确编程的情况下从数据中学习和改进。主要算法包括监督学习、无监督学习和强化学习。",
        "type": "技术文档",
        "category": "机器学习",
        "date": "2023-08-22",
        "relevance": 0.88,
        "views": 980
    },
    {
        "id": 3,
        "title": "深度学习在自然语言处理中的突破",
        "content": "深度学习技术特别是Transformer架构，彻底改变了自然语言处理领域，在机器翻译、文本生成和情感分析等任务上取得了显著进展。",
        "type": "研究论文",
        "category": "深度学习",
        "date": "2023-11-30",
        "relevance": 0.82,
        "views": 750
    },
    {
        "id": 4,
        "title": "知识图谱构建与应用实践",
        "content": "知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。广泛应用于搜索引擎、智能问答和推荐系统等领域。",
        "type": "实践指南",
        "category": "知识图谱",
        "date": "2023-09-10",
        "relevance": 0.78,
        "views": 620
    },
    {
        "id": 5,
        "title": "大语言模型技术综述",
        "content": "大语言模型是基于Transformer架构的预训练模型，具有强大的语言理解和生成能力。ChatGPT、GPT-4等模型展示了在对话、创作和分析方面的卓越性能。",
        "type": "综述",
        "category": "大模型",
        "date": "2023-12-01",
        "relevance": 0.92,
        "views": 1500
    },
    {
        "id": 6,
        "title": "计算机视觉最新进展",
        "content": "计算机视觉使计算机能够从图像和视频中获取高层次的理解。最新进展包括目标检测、图像分割和生成式AI在视觉创作中的应用。",
        "type": "技术报告",
        "category": "计算机视觉",
        "date": "2023-10-18",
        "relevance": 0.75,
        "views": 520
    },
    {
        "id": 7,
        "title": "数据挖掘与商业智能",
        "content": "数据挖掘是从大量数据中发现模式、关联和异常的过程，为商业决策提供数据支持。常用技术包括聚类分析、关联规则挖掘和异常检测。",
        "type": "商业应用",
        "category": "数据挖掘",
        "date": "2023-07-25",
        "relevance": 0.70,
        "views": 480
    }
]

# 问答知识库
QA_KNOWLEDGE_BASE = {
    # 人工智能相关问题
    "什么是人工智能": {
        "answer": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行需要人类智能的任务的智能系统。这些任务包括学习、推理、问题解决、感知和语言理解。",
        "category": "人工智能",
        "confidence": 0.98
    },
    "人工智能有哪些应用领域": {
        "answer": "人工智能的主要应用领域包括：1. 自然语言处理（机器翻译、智能客服）；2. 计算机视觉（人脸识别、自动驾驶）；3. 推荐系统（电商、视频平台）；4. 医疗健康（疾病诊断、药物研发）；5. 金融科技（风险控制、智能投顾）。",
        "category": "人工智能",
        "confidence": 0.95
    },

    # 机器学习相关问题
    "机器学习是什么": {
        "answer": "机器学习是人工智能的一个子领域，使计算机系统能够从数据中学习和改进经验，而无需明确编程。主要类型包括监督学习、无监督学习和强化学习。",
        "category": "机器学习",
        "confidence": 0.97
    },
    "机器学习和深度学习的区别": {
        "answer": "机器学习和深度学习的主要区别在于：1. 机器学习通常需要特征工程，而深度学习可以自动学习特征；2. 深度学习使用多层神经网络，适合处理大规模复杂数据；3. 机器学习算法相对简单，深度学习模型更复杂但效果更好。",
        "category": "机器学习",
        "confidence": 0.90
    },

    # 知识图谱相关问题
    "知识图谱的作用": {
        "answer": "知识图谱的主要作用包括：1. 语义搜索：理解用户查询意图；2. 智能问答：提供准确答案；3. 推荐系统：基于实体关系推荐；4. 决策支持：提供结构化知识辅助决策。",
        "category": "知识图谱",
        "confidence": 0.96
    },
    "如何构建知识图谱": {
        "answer": "构建知识图谱的主要步骤：1. 知识获取（从文本、数据库等提取）；2. 知识表示（实体识别、关系抽取）；3. 知识存储（图数据库如Neo4j）；4. 知识应用（搜索、推理、可视化）。",
        "category": "知识图谱",
        "confidence": 0.88
    },

    # 自然语言处理相关问题
    "自然语言处理有哪些应用": {
        "answer": "自然语言处理（NLP）的应用包括：1. 机器翻译（如Google翻译）；2. 情感分析（评论情感判断）；3. 文本摘要（自动生成摘要）；4. 智能对话（ChatGPT等）；5. 信息提取（从文本提取结构化信息）。",
        "category": "自然语言处理",
        "confidence": 0.94
    },

    # 计算机视觉相关问题
    "计算机视觉是什么": {
        "answer": "计算机视觉是人工智能的一个分支，使计算机能够从图像和视频中获取高层次的理解。主要任务包括图像分类、目标检测、图像分割和人脸识别等。",
        "category": "计算机视觉",
        "confidence": 0.93
    },

    # 大模型相关问题
    "什么是大语言模型": {
        "answer": "大语言模型是基于Transformer架构的预训练模型，具有数十亿甚至数万亿参数。它们能够理解和生成自然语言文本，在对话、创作、翻译等任务上表现出色，如GPT系列、BERT等。",
        "category": "大模型",
        "confidence": 0.96
    }
}

# 系统统计信息
SYSTEM_STATS = {
    "data_volume": "5万+文档",
    "entity_count": "20万+实体",
    "relation_count": "50万+关系",
    "daily_queries": "1,500次",
    "response_time": "0.2秒",
    "accuracy_rate": "98.5%",
    "user_count": "2,300+",
    "update_time": "2023-12-25"
}


# ========== 路由定义 ==========

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """API状态检查"""
    return jsonify({
        "status": "running",
        "service": "数据可视化及信息检索与问答系统",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/api/status - 系统状态",
            "/api/graph - 知识图谱数据",
            "/api/search - 信息检索",
            "/api/qa - 智能问答",
            "/api/stats - 系统统计"
        ]
    })


@app.route('/api/graph')
def get_knowledge_graph():
    """获取知识图谱数据"""
    return jsonify(KNOWLEDGE_GRAPH)


@app.route('/api/graph/stats')
def get_graph_stats():
    """获取图谱统计信息"""
    nodes_by_type = {}
    for node in KNOWLEDGE_GRAPH["nodes"]:
        node_type = node["type"]
        nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1

    return jsonify({
        "total_nodes": len(KNOWLEDGE_GRAPH["nodes"]),
        "total_edges": len(KNOWLEDGE_GRAPH["edges"]),
        "nodes_by_type": nodes_by_type,
        "node_groups": len(set(node["group"] for node in KNOWLEDGE_GRAPH["nodes"])),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/search', methods=['POST'])
def search_documents():
    """信息检索"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip().lower()

        if not query:
            # 返回所有文档
            return jsonify({
                "query": "",
                "total": len(SEARCH_DOCUMENTS),
                "results": SEARCH_DOCUMENTS[:5],
                "search_time": 0.15
            })

        # 简单搜索匹配（模拟）
        results = []
        for doc in SEARCH_DOCUMENTS:
            # 计算匹配分数
            score = 0
            if query in doc['title'].lower():
                score += 0.5
            if query in doc['content'].lower():
                score += 0.3
            if query in doc['category'].lower():
                score += 0.2

            if score > 0:
                doc_copy = doc.copy()
                doc_copy['match_score'] = min(score, 0.95)  # 限制最高分
                results.append(doc_copy)

        # 按匹配度排序
        results.sort(key=lambda x: x['match_score'], reverse=True)

        # 模拟搜索时间
        search_time = round(0.1 + random.random() * 0.3, 3)

        return jsonify({
            "query": query,
            "total": len(results),
            "results": results[:10],
            "search_time": search_time
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "query": "",
            "total": 0,
            "results": [],
            "search_time": 0
        })


@app.route('/api/qa', methods=['POST'])
def qa_system():
    """智能问答"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({
                "question": "",
                "answer": "请提出问题。",
                "confidence": 0.0,
                "category": "系统提示"
            })

        # 检查是否有完全匹配的问题
        if question in QA_KNOWLEDGE_BASE:
            response = QA_KNOWLEDGE_BASE[question]
            return jsonify({
                "question": question,
                "answer": response["answer"],
                "confidence": response["confidence"],
                "category": response["category"]
            })

        # 关键词匹配
        matched_questions = []
        for q in QA_KNOWLEDGE_BASE:
            # 简单关键词匹配
            question_words = set(question.lower().replace('?', '').replace('？', '').split())
            q_words = set(q.lower().replace('?', '').replace('？', '').split())

            common_words = question_words.intersection(q_words)
            if len(common_words) >= 1:
                matched_questions.append((q, len(common_words)))

        if matched_questions:
            # 找到最佳匹配
            matched_questions.sort(key=lambda x: x[1], reverse=True)
            best_match = matched_questions[0][0]
            response = QA_KNOWLEDGE_BASE[best_match]

            # 根据匹配程度调整置信度
            confidence = max(0.6, response["confidence"] * 0.9)

            return jsonify({
                "question": question,
                "answer": response["answer"],
                "confidence": confidence,
                "category": response["category"]
            })

        # 如果没有匹配，根据问题类型生成回答
        question_lower = question.lower()
        if any(word in question_lower for word in ['人工', '智能', 'ai']):
            answer = "关于人工智能的问题，我可以告诉您：人工智能是模拟人类智能的技术，包括机器学习、自然语言处理、计算机视觉等多个子领域。"
            category = "人工智能"
        elif any(word in question_lower for word in ['学习', '训练', '模型']):
            answer = "关于机器学习的问题，机器学习是通过算法让计算机从数据中学习规律，包括监督学习、无监督学习和强化学习等类型。"
            category = "机器学习"
        elif any(word in question_lower for word in ['知识', '图谱', 'kg']):
            answer = "关于知识图谱的问题，知识图谱是结构化的语义知识库，用于描述现实世界中的实体及其关系。"
            category = "知识图谱"
        elif any(word in question_lower for word in ['语言', '文本', '翻译']):
            answer = "关于自然语言处理的问题，NLP是人工智能的重要分支，研究计算机与人类语言之间的交互。"
            category = "自然语言处理"
        else:
            answer = "这个问题超出了我当前的知识范围。请尝试询问关于人工智能、机器学习、知识图谱、自然语言处理或计算机视觉相关的问题。"
            category = "未知"

        return jsonify({
            "question": question,
            "answer": answer,
            "confidence": 0.5,
            "category": category
        })

    except Exception as e:
        return jsonify({
            "question": "",
            "answer": f"处理问题时发生错误：{str(e)}",
            "confidence": 0.0,
            "category": "系统错误"
        })


@app.route('/api/stats')
def get_system_stats():
    """获取系统统计信息"""
    current_stats = SYSTEM_STATS.copy()
    current_stats["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_stats["system_uptime"] = "99.8%"
    current_stats["active_sessions"] = random.randint(50, 150)

    # 生成一些随机数据用于展示
    current_stats["weekly_queries"] = random.randint(8000, 12000)
    current_stats["monthly_users"] = random.randint(1500, 2500)
    current_stats["data_growth"] = "5%每月"

    return jsonify(current_stats)


@app.route('/api/search/suggest', methods=['GET'])
def search_suggest():
    """搜索建议"""
    query = request.args.get('q', '').lower()

    if not query:
        return jsonify([])

    suggestions = []
    for doc in SEARCH_DOCUMENTS:
        if query in doc['title'].lower() or query in doc['category'].lower():
            suggestions.append({
                "text": doc['title'],
                "category": doc['category']
            })

    # 添加一些通用建议
    if query.startswith('人工'):
        suggestions.append({"text": "人工智能发展历程", "category": "人工智能"})
    elif query.startswith('机器'):
        suggestions.append({"text": "机器学习算法", "category": "机器学习"})
    elif query.startswith('深度'):
        suggestions.append({"text": "深度学习应用", "category": "深度学习"})

    # 去重
    unique_suggestions = []
    seen = set()
    for suggestion in suggestions:
        key = suggestion['text']
        if key not in seen:
            unique_suggestions.append(suggestion)
            seen.add(key)

    return jsonify(unique_suggestions[:5])


@app.route('/api/qa/history', methods=['GET'])
def get_qa_history():
    """获取问答历史（模拟）"""
    history = [
        {
            "question": "什么是人工智能？",
            "answer": QA_KNOWLEDGE_BASE["什么是人工智能"]["answer"],
            "time": "10:30",
            "confidence": 0.98
        },
        {
            "question": "机器学习的应用有哪些？",
            "answer": "机器学习广泛应用于图像识别、语音识别、推荐系统、金融风控、医疗诊断等领域。",
            "time": "10:35",
            "confidence": 0.85
        },
        {
            "question": "知识图谱如何构建？",
            "answer": QA_KNOWLEDGE_BASE["如何构建知识图谱"]["answer"],
            "time": "10:40",
            "confidence": 0.88
        }
    ]
    return jsonify(history)


# ========== 启动应用 ==========

if __name__ == '__main__':
    # 确保模板目录存在
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("=" * 60)
    print("数据可视化及信息检索与问答系统 V1.0")
    print("=" * 60)
    print("系统正在启动...")
    print(f"服务地址: http://127.0.0.1:5000")
    print(f"API地址: http://127.0.0.1:5000/api/status")
    print("-" * 60)
    print("可用端点:")
    print("  /              - 主界面")
    print("  /api/status    - 系统状态")
    print("  /api/graph     - 知识图谱数据")
    print("  /api/search    - 信息检索")
    print("  /api/qa        - 智能问答")
    print("  /api/stats     - 系统统计")
    print("=" * 60)

    app.run(debug=True, host='127.0.0.1', port=5000)