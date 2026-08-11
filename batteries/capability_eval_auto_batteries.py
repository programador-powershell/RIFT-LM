#!/usr/bin/env python3
"""CAP — Bateria de capacidades estilo OpenRouter compare (docs/C3_CONTRACTS_V1.md §9).

Probe leve embutido — não é MMLU/HumanEval/SWE-bench completos. Sete categorias,
sete registros POR MODELO (avaliam o modelo baseline, não uma tecnologia de otimização):

    CAP_INTELLIGENCE   20 questões de múltipla escolha embutidas (A-D), greedy;
                       score = % de acertos.
    CAP_CODING         8 completações de função Python com asserts embutidos,
                       executadas em sandbox restrito (sem import/IO, watchdog 5s);
                       score = % de tarefas com TODOS os asserts passando.
    CAP_AGENTIC        8 tarefas de function-calling: gerar APENAS uma chamada JSON
                       {"name": ..., "arguments": {...}} contra schema embutido;
                       pontos por tarefa: JSON parseável (50) + nome correto (25) +
                       chaves obrigatórias presentes (25); score = média dos pontos.
    CAP_DEEPSEARCH_QA  8 tarefas de QA multi-hop (docs/C3_CONTRACTS_V1.md §15):
                       mini-corpus de 2-3 passagens fictícias PT-BR/EN no prompt;
                       a resposta exige combinar fatos entre passagens; score = %
                       de acertos por match normalizado (casefold, sem acentos e
                       sem pontuação). Inspirado em DeepSearch QA.
    CAP_MCP_ATLAS      8 tarefas de uso de ferramentas estilo MCP: 3-4 tool
                       schemas por prompt (2 cenários multi-passo com o resultado
                       do passo 1 embutido); pontos: JSON parseável (40) +
                       ferramenta correta (30) + argumentos obrigatórios com
                       valores plausíveis (30). Inspirado em MCP Atlas.
    CAP_TAU3_BENCH     8 tarefas de agente de atendimento com política declarada
                       (domínios bancário/aéreo; às vezes o correto é RECUSAR ou
                       escalar); pontos: ação correta (60) + conformidade com a
                       política (40); mesma convenção de chamada JSON. Inspirado
                       em τ³-Bench.
    CAP_SWE_BENCH      8 reparos de código: função Python com bug + teste
                       falhando + erro no prompt; o modelo emite a função
                       corrigida, executada no MESMO sandbox restrito do
                       CAP_CODING; score = % de tarefas com toda a suíte de
                       asserts passando. Inspirado em SWE-Bench.

Contrato (§9): technology="CAP", benchmark_protocol="CAPABILITY_PROBE_V1",
comparison_role=null (NUNCA entra na seleção do winner), eligible_for_primary_ranking=false,
quality=null, tok_s/RAM/disco de topo nulos; score 0-100 em metrics.capability.score.
Publicação: POST /api/results com HTTPS obrigatório + RIFT_INGEST_TOKEN >= 32 caracteres.

Backends de geração (--backend, default transformers — sem mudança de
comportamento no default):
    transformers  model.generate greedy local (torch/transformers exigidos);
    llamacpp      HTTP no llama-server (--server-url; tenta /completion nativo e
                  /v1/completions OpenAI-compatível), greedy; mesmas tarefas e
                  pontuação; torch/transformers NÃO são exigidos. model_id dos
                  registros pode ser rotulado com --model-id-label (ex.:
                  "unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL", contrato §11).

Uso:
    python capability_eval_auto_batteries.py --model Qwen/Qwen2.5-0.5B
    python capability_eval_auto_batteries.py --backend llamacpp \
        --server-url http://127.0.0.1:8090 \
        --model unsloth/Muse-Glimmer-30B-GGUF \
        --model-id-label "unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL"

Sem pip install automático: dependências ausentes geram SystemExit com instruções.
Sai com código 0 mesmo com registros FAIL; código != 0 apenas em crash antes de
qualquer registro ser gravado.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import threading
import traceback
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# --- dependências (SEM pip automático: o launcher Colab instala antes) ---
# torch/transformers são exigidos APENAS no backend padrão (transformers); o
# SystemExit é adiado para main(), depois de conhecer --backend: no backend
# llamacpp a geração acontece via HTTP no llama-server e nada disso é preciso.
_MISSING_DEPS: List[str] = []
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
    _MISSING_DEPS.append("torch")
try:
    import transformers  # noqa: F401  (usado em load_model)
except ImportError:
    _MISSING_DEPS.append("transformers")

MISSING_DEPS_MESSAGE = (
    "[CAP] Dependências ausentes: " + ", ".join(_MISSING_DEPS) + ". "
    "Este script NÃO instala pacotes automaticamente — o launcher Colab deveria "
    "tê-los instalado. Instale manualmente: pip install torch transformers "
    "accelerate sentencepiece (versões pinadas conforme o launcher). "
    "Alternativa sem torch: --backend llamacpp --server-url http://127.0.0.1:8090."
) if _MISSING_DEPS else ""

BENCHMARK_PROTOCOL = "CAPABILITY_PROBE_V1"
HONEST_LABEL = "probe leve embutido — não é MMLU/HumanEval/SWE-bench completos"
DEFAULT_ENDPOINT = "https://rift-lm.vercel.app/api/results"
CODING_TIMEOUT_S = 5.0

# Contador global de registros gravados: exit != 0 só em crash ANTES do primeiro.
EMITTED_RECORDS = 0


# ---------------------------------------------------------------------------
# Tarefas embutidas e determinísticas — CAP_INTELLIGENCE (20 questões A-D)
# Mix PT-BR/EN, respostas inequívocas (conhecimentos gerais/lógica/matemática).
# ---------------------------------------------------------------------------

INTELLIGENCE_TASKS: List[Dict[str, Any]] = [
    {"id": "int_01", "question": "Quanto é 7 x 8?",
     "options": {"A": "54", "B": "56", "C": "58", "D": "64"}, "answer": "B"},
    {"id": "int_02", "question": "Qual é a capital da França?",
     "options": {"A": "Paris", "B": "Roma", "C": "Madri", "D": "Londres"}, "answer": "A"},
    {"id": "int_03", "question": "What is the chemical symbol for water?",
     "options": {"A": "CO2", "B": "H2O", "C": "O2", "D": "NaCl"}, "answer": "B"},
    {"id": "int_04", "question": "Qual planeta é conhecido como Planeta Vermelho?",
     "options": {"A": "Vênus", "B": "Júpiter", "C": "Saturno", "D": "Marte"}, "answer": "D"},
    {"id": "int_05", "question": "Quanto é 15% de 200?",
     "options": {"A": "30", "B": "15", "C": "45", "D": "20"}, "answer": "A"},
    {"id": "int_06", "question": "Which number completes the sequence 2, 4, 8, 16, ...?",
     "options": {"A": "18", "B": "24", "C": "32", "D": "20"}, "answer": "C"},
    {"id": "int_07", "question": "Qual é o maior oceano da Terra?",
     "options": {"A": "Atlântico", "B": "Índico", "C": "Ártico", "D": "Pacífico"}, "answer": "D"},
    {"id": "int_08", "question": "Se todos os gatos são mamíferos e Tom é um gato, então Tom é:",
     "options": {"A": "um réptil", "B": "um mamífero", "C": "um pássaro", "D": "um peixe"}, "answer": "B"},
    {"id": "int_09", "question": "What is the square root of 144?",
     "options": {"A": "12", "B": "14", "C": "10", "D": "11"}, "answer": "A"},
    {"id": "int_10", "question": "Quantos lados tem um hexágono?",
     "options": {"A": "5", "B": "6", "C": "7", "D": "8"}, "answer": "B"},
    {"id": "int_11", "question": "Em que continente fica o Egito?",
     "options": {"A": "Ásia", "B": "Europa", "C": "África", "D": "Oceania"}, "answer": "C"},
    {"id": "int_12", "question": "Which gas do plants absorb from the atmosphere during photosynthesis?",
     "options": {"A": "Oxygen", "B": "Nitrogen", "C": "Carbon dioxide", "D": "Helium"}, "answer": "C"},
    {"id": "int_13", "question": "Quanto é 100 dividido por 4?",
     "options": {"A": "20", "B": "40", "C": "24", "D": "25"}, "answer": "D"},
    {"id": "int_14", "question": "Quem escreveu 'Dom Casmurro'?",
     "options": {"A": "José de Alencar", "B": "Machado de Assis",
                 "C": "Carlos Drummond de Andrade", "D": "Clarice Lispector"}, "answer": "B"},
    {"id": "int_15", "question": "How many minutes are there in two hours?",
     "options": {"A": "60", "B": "90", "C": "100", "D": "120"}, "answer": "D"},
    {"id": "int_16", "question": "Qual destes números é primo?",
     "options": {"A": "9", "B": "15", "C": "17", "D": "21"}, "answer": "C"},
    {"id": "int_17", "question": "Ao nível do mar, a água ferve a:",
     "options": {"A": "90 graus Celsius", "B": "100 graus Celsius",
                 "C": "110 graus Celsius", "D": "120 graus Celsius"}, "answer": "B"},
    {"id": "int_18", "question": "Which is the largest planet in the Solar System?",
     "options": {"A": "Jupiter", "B": "Mars", "C": "Earth", "D": "Venus"}, "answer": "A"},
    {"id": "int_19", "question": "Se um trem parte às 14h00 e chega às 16h30, quanto durou a viagem?",
     "options": {"A": "1h30", "B": "2h00", "C": "2h30", "D": "3h00"}, "answer": "C"},
    {"id": "int_20", "question": "Quanto é 3 ao quadrado mais 4 ao quadrado?",
     "options": {"A": "25", "B": "24", "C": "49", "D": "12"}, "answer": "A"},
]


# ---------------------------------------------------------------------------
# Tarefas embutidas — CAP_CODING (8 completações de função + asserts)
# O header (assinatura + docstring) é dado ao modelo; o corpo gerado é executado
# em sandbox restrito. Asserts são expressões booleanas avaliadas no sandbox.
# ---------------------------------------------------------------------------

CODING_TASKS: List[Dict[str, Any]] = [
    {
        "id": "cod_fatorial", "name": "fatorial",
        "header": 'def fatorial(n):\n    """Retorna o fatorial de n (inteiro n >= 0). Ex.: fatorial(4) -> 24."""\n',
        "asserts": ["fatorial(0) == 1", "fatorial(1) == 1", "fatorial(5) == 120", "fatorial(7) == 5040"],
    },
    {
        "id": "cod_inverter_string", "name": "inverter_string",
        "header": 'def inverter_string(s):\n    """Retorna a string s invertida. Ex.: inverter_string("abc") -> "cba"."""\n',
        "asserts": ['inverter_string("abc") == "cba"', 'inverter_string("") == ""',
                    'inverter_string("radar") == "radar"', 'inverter_string("Python") == "nohtyP"'],
    },
    {
        "id": "cod_fizzbuzz", "name": "fizzbuzz",
        "header": ('def fizzbuzz(n):\n    """Retorna "FizzBuzz" se n for divisível por 3 e por 5, "Fizz" se apenas\n'
                   '    por 3, "Buzz" se apenas por 5; caso contrário retorna str(n)."""\n'),
        "asserts": ['fizzbuzz(3) == "Fizz"', 'fizzbuzz(10) == "Buzz"', 'fizzbuzz(15) == "FizzBuzz"',
                    'fizzbuzz(7) == "7"', 'fizzbuzz(45) == "FizzBuzz"'],
    },
    {
        "id": "cod_soma_pares", "name": "soma_pares",
        "header": ('def soma_pares(numeros):\n    """Retorna a soma dos números pares da lista.\n'
                   '    Ex.: soma_pares([1, 2, 3, 4]) -> 6."""\n'),
        "asserts": ["soma_pares([1, 2, 3, 4]) == 6", "soma_pares([]) == 0",
                    "soma_pares([1, 3, 5]) == 0", "soma_pares([2, 4, 6, 8]) == 20"],
    },
    {
        "id": "cod_eh_palindromo", "name": "eh_palindromo",
        "header": ('def eh_palindromo(s):\n    """Retorna True se s for palíndromo, ignorando maiúsculas/minúsculas.\n'
                   '    Ex.: eh_palindromo("Arara") -> True."""\n'),
        "asserts": ['eh_palindromo("arara")', 'not eh_palindromo("python")', 'eh_palindromo("Ana")',
                    'eh_palindromo("abba")', 'not eh_palindromo("abc")'],
    },
    {
        "id": "cod_maximo_lista", "name": "maximo_lista",
        "header": ('def maximo_lista(numeros):\n    """Retorna o maior número da lista (a lista nunca é vazia).\n'
                   '    Ex.: maximo_lista([3, 1, 2]) -> 3."""\n'),
        "asserts": ["maximo_lista([3, 1, 2]) == 3", "maximo_lista([-5, -2, -9]) == -2",
                    "maximo_lista([7]) == 7", "maximo_lista([1, 2, 10, 4]) == 10"],
    },
    {
        "id": "cod_contar_vogais", "name": "contar_vogais",
        "header": ('def contar_vogais(s):\n    """Retorna o número de vogais (a, e, i, o, u — maiúsculas ou\n'
                   '    minúsculas, sem acento) presentes em s."""\n'),
        "asserts": ['contar_vogais("banana") == 3', 'contar_vogais("xyz") == 0',
                    'contar_vogais("AeIoU") == 5', 'contar_vogais("chatbot") == 2'],
    },
    {
        "id": "cod_fibonacci", "name": "fibonacci",
        "header": ('def fibonacci(n):\n    """Retorna o n-ésimo número de Fibonacci:\n'
                   '    fibonacci(0) -> 0, fibonacci(1) -> 1, fibonacci(10) -> 55."""\n'),
        "asserts": ["fibonacci(0) == 0", "fibonacci(1) == 1", "fibonacci(2) == 1", "fibonacci(10) == 55"],
    },
]


# ---------------------------------------------------------------------------
# Tarefas embutidas — CAP_AGENTIC (8 chamadas de ferramenta em JSON)
# Espelha a disciplina de "JSON floor" da spec GEYSER: a saída precisa ser um
# JSON 100% parseável para pontuar a primeira metade.
# ---------------------------------------------------------------------------

AGENTIC_TASKS: List[Dict[str, Any]] = [
    {
        "id": "agt_agendar_reuniao",
        "tool": {"name": "agendar_reuniao", "description": "Agenda uma reunião no calendário.",
                 "parameters": {"pessoa": "string", "data": "string", "hora": "string"},
                 "required": ["pessoa", "data", "hora"]},
        "instruction": "Agende uma reunião com Ana amanhã às 15h.",
    },
    {
        "id": "agt_enviar_email",
        "tool": {"name": "enviar_email", "description": "Envia um e-mail para um destinatário.",
                 "parameters": {"destinatario": "string", "assunto": "string", "corpo": "string"},
                 "required": ["destinatario", "assunto"]},
        "instruction": "Envie um e-mail para joao@exemplo.com com o assunto 'Relatório mensal'.",
    },
    {
        "id": "agt_clima_atual",
        "tool": {"name": "clima_atual", "description": "Consulta o clima atual de uma cidade.",
                 "parameters": {"cidade": "string"},
                 "required": ["cidade"]},
        "instruction": "Qual é o clima agora em Porto Alegre?",
    },
    {
        "id": "agt_criar_tarefa",
        "tool": {"name": "criar_tarefa", "description": "Cria uma tarefa na lista de afazeres.",
                 "parameters": {"titulo": "string", "prioridade": "string"},
                 "required": ["titulo", "prioridade"]},
        "instruction": "Crie uma tarefa chamada 'Revisar contrato' com prioridade alta.",
    },
    {
        "id": "agt_converter_moeda",
        "tool": {"name": "converter_moeda", "description": "Converte um valor entre duas moedas.",
                 "parameters": {"valor": "number", "de": "string", "para": "string"},
                 "required": ["valor", "de", "para"]},
        "instruction": "Converta 250 dólares americanos para reais.",
    },
    {
        "id": "agt_definir_alarme",
        "tool": {"name": "definir_alarme", "description": "Define um alarme no relógio.",
                 "parameters": {"hora": "string", "rotulo": "string"},
                 "required": ["hora"]},
        "instruction": "Defina um alarme para as 6h30 de amanhã.",
    },
    {
        "id": "agt_buscar_voo",
        "tool": {"name": "buscar_voo", "description": "Busca voos entre duas cidades em uma data.",
                 "parameters": {"origem": "string", "destino": "string", "data": "string"},
                 "required": ["origem", "destino", "data"]},
        "instruction": "Busque um voo de São Paulo para Recife no dia 2026-09-01.",
    },
    {
        "id": "agt_tocar_musica",
        "tool": {"name": "tocar_musica", "description": "Toca uma música de um artista.",
                 "parameters": {"artista": "string", "faixa": "string"},
                 "required": ["artista"]},
        "instruction": "Toque uma música da banda Legião Urbana.",
    },
]


# ---------------------------------------------------------------------------
# Tarefas embutidas — CAP_DEEPSEARCH_QA (8 tarefas de QA multi-hop, §15)
# Mini-corpus de 2-3 passagens por prompt sobre entidades INVENTADAS (evita
# memorização); a resposta exige combinar fatos entre passagens. Aliases em
# "answers" cobrem grafias equivalentes; o match é normalizado (casefold,
# sem acentos, sem pontuação, delimitado por palavra).
# ---------------------------------------------------------------------------

DEEPSEARCH_TASKS: List[Dict[str, Any]] = [
    {
        "id": "dsq_zorvatek",
        "passages": [
            "A Zorvatek, fabricante fictícia de sensores industriais, foi fundada em 1987 "
            "em Curitiba por Helena Vasques.",
            "Onze anos após a sua fundação, a Zorvatek transferiu a sede para Florianópolis.",
            "A Brimoltec, concorrente da Zorvatek, foi fundada em 1990 e mantém a sede em Recife.",
        ],
        "question": "Em que ano a Zorvatek transferiu a sede para Florianópolis?",
        "answers": ["1998"],
    },
    {
        "id": "dsq_bramtec",
        "passages": [
            "A Bramtec Alimentos foi criada por Kellan Moraes, que também dirige o "
            "instituto fictício Ondaviva.",
            "Kellan Moraes nasceu em Ouro Preto e formou-se em engenharia de alimentos.",
        ],
        "question": "Em que cidade nasceu o fundador da Bramtec Alimentos?",
        "answers": ["Ouro Preto"],
    },
    {
        "id": "dsq_vantorix",
        "passages": [
            "Em 2019 a Vantorix Logística empregava 120 funcionários.",
            "Em 2020 a Vantorix dobrou o seu quadro de funcionários em relação a 2019.",
            "A Vantorix atua no transporte fluvial da região do rio fictício Iberama.",
        ],
        "question": "Quantos funcionários a Vantorix tinha em 2020?",
        "answers": ["240"],
    },
    {
        "id": "dsq_skyvenna",
        "passages": [
            "Skyvenna is a fictional regional airline operating exactly 14 routes.",
            "Half of Skyvenna's routes depart from its hub in Porto Novo.",
        ],
        "question": "How many Skyvenna routes depart from Porto Novo?",
        "answers": ["7", "seven"],
    },
    {
        "id": "dsq_lumio",
        "passages": [
            "O tablet fictício Lumio X custa R$ 320 na loja oficial.",
            "O Lumio X Pro custa exatamente R$ 130 a mais que o Lumio X.",
            "O carregador do Lumio X é vendido separadamente por R$ 45.",
        ],
        "question": "Quanto custa o Lumio X Pro, em reais?",
        "answers": ["450"],
    },
    {
        "id": "dsq_belverde",
        "passages": [
            "Belverde é a capital do estado imaginário de Alvorada.",
            "O rio fictício Tarumeu corta a cidade de Belverde de norte a sul.",
            "O estado de Alvorada tem outras duas cidades grandes: Rocha Clara e Pontal do Sol.",
        ],
        "question": "Qual rio corta a capital do estado de Alvorada?",
        "answers": ["Tarumeu"],
    },
    {
        "id": "dsq_quenda",
        "passages": [
            "Marlowe Finch is the chief executive of the fictional biotech Quenda Labs.",
            "Before joining Quenda Labs, its chief executive worked at Ostrix for nine years.",
            "Quenda Labs was founded in 2003 in the invented city of Greyhaven.",
        ],
        "question": "For how many years did Marlowe Finch work at Ostrix?",
        "answers": ["nine", "9"],
    },
    {
        "id": "dsq_sonora",
        "passages": [
            "O festival fictício Sonora acontece todos os anos no mês de março.",
            "Na edição de 2024, o Sonora foi realizado na cidade imaginária de Pedra Verde.",
        ],
        "question": "Em que cidade aconteceu, em 2024, o festival que ocorre sempre em março?",
        "answers": ["Pedra Verde"],
    },
]


# ---------------------------------------------------------------------------
# Tarefas embutidas — CAP_MCP_ATLAS (8 tarefas de uso de ferramentas MCP, §15)
# Cada prompt lista 3-4 tool schemas estilo MCP (name/description/inputSchema);
# o modelo escolhe a ferramenta certa e emite a chamada JSON. Dois cenários são
# multi-passo: "step_result" embute o retorno do passo 1 e o esperado é a
# SEGUNDA chamada. "expect" traz dicas normalizadas de valores plausíveis por
# argumento (basta uma dica presente no valor normalizado).
# ---------------------------------------------------------------------------

MCP_TOOLSET_PEDIDOS: List[Dict[str, Any]] = [
    {"name": "consultar_pedido", "description": "Retorna o pedido mais recente de um cliente.",
     "parameters": {"cliente_id": "string"}, "required": ["cliente_id"]},
    {"name": "rastrear_entrega", "description": "Rastreia a entrega de um pedido pelo id.",
     "parameters": {"pedido_id": "string"}, "required": ["pedido_id"]},
    {"name": "cancelar_pedido", "description": "Cancela um pedido existente.",
     "parameters": {"pedido_id": "string", "motivo": "string"}, "required": ["pedido_id"]},
    {"name": "listar_produtos", "description": "Lista os produtos de uma categoria.",
     "parameters": {"categoria": "string"}, "required": ["categoria"]},
]

MCP_TOOLSET_FINANCAS: List[Dict[str, Any]] = [
    {"name": "cotacao_moeda", "description": "Retorna a cotação entre duas moedas.",
     "parameters": {"de": "string", "para": "string"}, "required": ["de", "para"]},
    {"name": "saldo_conta", "description": "Consulta o saldo de uma conta.",
     "parameters": {"conta_id": "string"}, "required": ["conta_id"]},
    {"name": "extrato_conta", "description": "Retorna o extrato mensal de uma conta.",
     "parameters": {"conta_id": "string", "mes": "string"}, "required": ["conta_id"]},
    {"name": "calcular_juros", "description": "Calcula juros simples de um principal.",
     "parameters": {"principal": "number", "taxa_anual": "number", "meses": "number"},
     "required": ["principal", "taxa_anual", "meses"]},
]

MCP_TOOLSET_VIAGEM: List[Dict[str, Any]] = [
    {"name": "buscar_hotel", "description": "Busca hotéis em uma cidade para um período.",
     "parameters": {"cidade": "string", "checkin": "string", "checkout": "string"},
     "required": ["cidade", "checkin"]},
    {"name": "reservar_quarto", "description": "Reserva um quarto em um hotel pelo id.",
     "parameters": {"hotel_id": "string", "quarto": "string"}, "required": ["hotel_id"]},
    {"name": "clima_cidade", "description": "Consulta a previsão do tempo de uma cidade.",
     "parameters": {"cidade": "string"}, "required": ["cidade"]},
]

MCP_ATLAS_TASKS: List[Dict[str, Any]] = [
    {
        "id": "mcp_rastrear",
        "tools": MCP_TOOLSET_PEDIDOS,
        "instruction": "Rastreie a entrega do pedido PED-4410.",
        "expected_tool": "rastrear_entrega",
        "expect": {"pedido_id": ["4410"]},
    },
    {
        "id": "mcp_listar_produtos",
        "tools": MCP_TOOLSET_PEDIDOS,
        "instruction": "Liste os produtos da categoria notebooks.",
        "expected_tool": "listar_produtos",
        "expect": {"categoria": ["notebooks", "notebook"]},
    },
    {
        "id": "mcp_consultar_pedido",
        "tools": MCP_TOOLSET_PEDIDOS,
        "instruction": "Qual é o pedido mais recente do cliente CLI-305?",
        "expected_tool": "consultar_pedido",
        "expect": {"cliente_id": ["305"]},
    },
    {
        # multi-passo: o resultado do passo 1 está no prompt; esperado é o passo 2
        "id": "mcp_cancelar_step2",
        "tools": MCP_TOOLSET_PEDIDOS,
        "step_result": ('Resultado do passo 1 — consultar_pedido({"cliente_id": "CLI-77"}) '
                        'retornou: {"pedido_id": "PED-9021", "status": "entregue_com_atraso"}'),
        "instruction": ("Cancele o pedido mais recente do cliente CLI-77 porque chegou "
                        "atrasado (o passo 1 acima já foi executado; emita a PRÓXIMA chamada)."),
        "expected_tool": "cancelar_pedido",
        "expect": {"pedido_id": ["9021"]},
    },
    {
        "id": "mcp_cotacao",
        "tools": MCP_TOOLSET_FINANCAS,
        "instruction": "Qual é a cotação do dólar americano para o real brasileiro?",
        "expected_tool": "cotacao_moeda",
        "expect": {"de": ["usd", "dolar", "dollar"], "para": ["brl", "real"]},
    },
    {
        "id": "mcp_juros",
        "tools": MCP_TOOLSET_FINANCAS,
        "instruction": ("Calcule os juros de um principal de 1000 com taxa anual de "
                        "12 por cento durante 6 meses."),
        "expected_tool": "calcular_juros",
        "expect": {"principal": ["1000"], "meses": ["6"]},
    },
    {
        "id": "mcp_buscar_hotel",
        "tools": MCP_TOOLSET_VIAGEM,
        "instruction": ("Encontre um hotel em Gramado com check-in em 2026-09-10 e "
                        "check-out em 2026-09-12."),
        "expected_tool": "buscar_hotel",
        "expect": {"cidade": ["gramado"], "checkin": ["2026 09 10", "10 09 2026"]},
    },
    {
        # multi-passo: o resultado do passo 1 está no prompt; esperado é o passo 2
        "id": "mcp_reservar_step2",
        "tools": MCP_TOOLSET_VIAGEM,
        "step_result": ('Resultado do passo 1 — buscar_hotel({"cidade": "Gramado"}) '
                        'retornou: {"hotel_id": "HTL-322", "nome": "Pousada Aurora"}'),
        "instruction": ("Reserve um quarto duplo no hotel encontrado em Gramado (o passo 1 "
                        "acima já foi executado; emita a PRÓXIMA chamada)."),
        "expected_tool": "reservar_quarto",
        "expect": {"hotel_id": ["322"]},
    },
]


# ---------------------------------------------------------------------------
# Tarefas embutidas — CAP_TAU3_BENCH (8 tarefas de agente com política, §15)
# Inspirado em τ³-Bench: política curta + solicitação do cliente + 3-4 ações
# (tool schemas). Em alguns casos o comportamento CORRETO é recusar ou escalar.
# Pontos: ação correta (60) + conformidade com a política (40) — a conformidade
# exige emitir uma ação VÁLIDA da lista que NÃO esteja proibida pela política
# para o caso ("forbidden_actions").
# ---------------------------------------------------------------------------

TAU3_ACTIONS_BANCO_REEMBOLSO: List[Dict[str, Any]] = [
    {"name": "processar_reembolso", "description": "Processa o reembolso de uma compra.",
     "parameters": {"compra_id": "string", "valor": "number"}, "required": ["compra_id"]},
    {"name": "escalar_para_humano", "description": "Encaminha o caso para aprovação humana.",
     "parameters": {"compra_id": "string", "motivo": "string"}, "required": ["compra_id"]},
    {"name": "recusar_solicitacao", "description": "Recusa a solicitação do cliente.",
     "parameters": {"compra_id": "string", "motivo": "string"}, "required": ["compra_id"]},
    {"name": "consultar_compra", "description": "Consulta os dados de uma compra.",
     "parameters": {"compra_id": "string"}, "required": ["compra_id"]},
]

TAU3_ACTIONS_BANCO_TRANSFERENCIA: List[Dict[str, Any]] = [
    {"name": "executar_transferencia", "description": "Executa uma transferência entre contas.",
     "parameters": {"conta_origem": "string", "conta_destino": "string", "valor": "number"},
     "required": ["conta_origem", "conta_destino", "valor"]},
    {"name": "escalar_para_humano", "description": "Encaminha o caso para verificação humana.",
     "parameters": {"conta_origem": "string", "motivo": "string"}, "required": ["conta_origem"]},
    {"name": "recusar_solicitacao", "description": "Recusa a solicitação do cliente.",
     "parameters": {"conta_origem": "string", "motivo": "string"}, "required": ["conta_origem"]},
]

TAU3_ACTIONS_AEREO: List[Dict[str, Any]] = [
    {"name": "remarcar_sem_taxa", "description": "Remarca o voo sem cobrança de taxa.",
     "parameters": {"reserva_id": "string", "nova_data": "string"}, "required": ["reserva_id"]},
    {"name": "remarcar_com_taxa", "description": "Remarca o voo cobrando a taxa de alteração.",
     "parameters": {"reserva_id": "string", "nova_data": "string"}, "required": ["reserva_id"]},
    {"name": "processar_reembolso_integral", "description": "Reembolsa 100% do valor da reserva.",
     "parameters": {"reserva_id": "string"}, "required": ["reserva_id"]},
    {"name": "recusar_solicitacao", "description": "Recusa a solicitação do passageiro.",
     "parameters": {"reserva_id": "string", "motivo": "string"}, "required": ["reserva_id"]},
]

TAU3_POLICY_REEMBOLSO = ("Reembolso é permitido somente até 24 horas após a compra; "
                         "reembolso acima de R$ 500 exige aprovação humana (escalar); "
                         "fora dessas condições, a solicitação deve ser recusada.")
TAU3_POLICY_TRANSFERENCIA = ("Transferências acima de R$ 10.000 exigem verificação humana "
                             "adicional (escalar); NUNCA execute uma transferência acima "
                             "desse limite sem verificação.")
TAU3_POLICY_AEREO = ("Remarcação gratuita apenas para tarifa Flex; tarifa Básica pode "
                     "remarcar somente pagando a taxa de alteração; reembolso integral "
                     "somente quando o voo foi cancelado pela companhia; caso contrário, "
                     "recusar o reembolso.")

TAU3_TASKS: List[Dict[str, Any]] = [
    {
        "id": "tau_reembolso_ok",
        "policy": TAU3_POLICY_REEMBOLSO,
        "actions": TAU3_ACTIONS_BANCO_REEMBOLSO,
        "request": ("Cliente pede reembolso da compra CMP-101, feita há 3 horas, "
                    "no valor de R$ 120."),
        "expected_action": "processar_reembolso",
        "forbidden_actions": ["recusar_solicitacao"],
    },
    {
        "id": "tau_reembolso_tarde",
        "policy": TAU3_POLICY_REEMBOLSO,
        "actions": TAU3_ACTIONS_BANCO_REEMBOLSO,
        "request": ("Cliente pede reembolso da compra CMP-202, feita há 3 dias, "
                    "no valor de R$ 100."),
        "expected_action": "recusar_solicitacao",
        "forbidden_actions": ["processar_reembolso"],
    },
    {
        "id": "tau_reembolso_alto",
        "policy": TAU3_POLICY_REEMBOLSO,
        "actions": TAU3_ACTIONS_BANCO_REEMBOLSO,
        "request": ("Cliente pede reembolso da compra CMP-303, feita há 2 horas, "
                    "no valor de R$ 900."),
        "expected_action": "escalar_para_humano",
        "forbidden_actions": ["processar_reembolso"],
    },
    {
        "id": "tau_reembolso_tarde_alto",
        "policy": TAU3_POLICY_REEMBOLSO,
        "actions": TAU3_ACTIONS_BANCO_REEMBOLSO,
        "request": ("Cliente pede reembolso da compra CMP-404, feita há 5 dias, "
                    "no valor de R$ 800."),
        "expected_action": "recusar_solicitacao",
        "forbidden_actions": ["processar_reembolso"],
    },
    {
        "id": "tau_transferencia_alta",
        "policy": TAU3_POLICY_TRANSFERENCIA,
        "actions": TAU3_ACTIONS_BANCO_TRANSFERENCIA,
        "request": ("Cliente da conta CT-88 pede transferência de R$ 25.000 para a "
                    "conta CT-99, sem verificação adicional feita."),
        "expected_action": "escalar_para_humano",
        "forbidden_actions": ["executar_transferencia"],
    },
    {
        "id": "tau_remarcar_flex",
        "policy": TAU3_POLICY_AEREO,
        "actions": TAU3_ACTIONS_AEREO,
        "request": ("Passageiro com tarifa Flex, reserva RSV-501, pede remarcação "
                    "do voo para 2026-10-05."),
        "expected_action": "remarcar_sem_taxa",
        "forbidden_actions": ["remarcar_com_taxa", "recusar_solicitacao"],
    },
    {
        "id": "tau_remarcar_basica",
        "policy": TAU3_POLICY_AEREO,
        "actions": TAU3_ACTIONS_AEREO,
        "request": ("Passageiro com tarifa Básica, reserva RSV-602, pede remarcação "
                    "SEM pagar taxa."),
        "expected_action": "remarcar_com_taxa",
        "forbidden_actions": ["remarcar_sem_taxa"],
    },
    {
        "id": "tau_reembolso_no_show",
        "policy": TAU3_POLICY_AEREO,
        "actions": TAU3_ACTIONS_AEREO,
        "request": ("Passageiro perdeu o voo (não compareceu), reserva RSV-703 na "
                    "tarifa Básica, e pede reembolso integral; o voo decolou normalmente."),
        "expected_action": "recusar_solicitacao",
        "forbidden_actions": ["processar_reembolso_integral"],
    },
]


# ---------------------------------------------------------------------------
# Tarefas embutidas — CAP_SWE_BENCH (8 reparos de código, §15)
# Função Python COM BUG + teste que falha + erro observado; o modelo emite a
# função corrigida. Reutiliza o sandbox/watchdog/extração do CAP_CODING
# ("header" = assinatura, para o fallback de corpo indentado).
# ---------------------------------------------------------------------------

SWE_TASKS: List[Dict[str, Any]] = [
    {
        "id": "swe_soma_lista", "name": "soma_lista",
        "header": "def soma_lista(numeros):\n",
        "buggy": ("def soma_lista(numeros):\n"
                  "    total = 0\n"
                  "    for i in range(len(numeros) - 1):\n"
                  "        total += numeros[i]\n"
                  "    return total\n"),
        "failing_test": "soma_lista([1, 2, 3]) == 6",
        "error": "AssertionError: soma_lista([1, 2, 3]) retornou 3, esperado 6",
        "asserts": ["soma_lista([1, 2, 3]) == 6", "soma_lista([]) == 0",
                    "soma_lista([5]) == 5", "soma_lista([2, 4, 6, 8]) == 20"],
    },
    {
        "id": "swe_media", "name": "media",
        "header": "def media(numeros):\n",
        "buggy": ("def media(numeros):\n"
                  "    return sum(numeros) / (len(numeros) + 1)\n"),
        "failing_test": "media([2, 4, 6]) == 4.0",
        "error": "AssertionError: media([2, 4, 6]) retornou 3.0, esperado 4.0",
        "asserts": ["media([2, 4, 6]) == 4.0", "media([5]) == 5.0",
                    "media([1, 2, 3, 4]) == 2.5"],
    },
    {
        "id": "swe_eh_par", "name": "eh_par",
        "header": "def eh_par(n):\n",
        "buggy": ("def eh_par(n):\n"
                  "    return n % 2 == 1\n"),
        "failing_test": "eh_par(4) == True",
        "error": "AssertionError: eh_par(4) retornou False, esperado True",
        "asserts": ["eh_par(4)", "not eh_par(7)", "eh_par(0)", "not eh_par(-3)"],
    },
    {
        "id": "swe_maior", "name": "maior",
        "header": "def maior(a, b):\n",
        "buggy": ("def maior(a, b):\n"
                  "    if a > b:\n"
                  "        return b\n"
                  "    return a\n"),
        "failing_test": "maior(3, 5) == 5",
        "error": "AssertionError: maior(3, 5) retornou 3, esperado 5",
        "asserts": ["maior(3, 5) == 5", "maior(10, 2) == 10",
                    "maior(-1, -5) == -1", "maior(7, 7) == 7"],
    },
    {
        "id": "swe_celsius_para_fahrenheit", "name": "celsius_para_fahrenheit",
        "header": "def celsius_para_fahrenheit(c):\n",
        "buggy": ("def celsius_para_fahrenheit(c):\n"
                  "    return c * 9 / 5 - 32\n"),
        "failing_test": "celsius_para_fahrenheit(0) == 32.0",
        "error": "AssertionError: celsius_para_fahrenheit(0) retornou -32.0, esperado 32.0",
        "asserts": ["celsius_para_fahrenheit(0) == 32.0", "celsius_para_fahrenheit(100) == 212.0",
                    "celsius_para_fahrenheit(10) == 50.0", "celsius_para_fahrenheit(-40) == -40.0"],
    },
    {
        "id": "swe_primeiro_ultimo", "name": "primeiro_ultimo",
        "header": "def primeiro_ultimo(lista):\n",
        "buggy": ("def primeiro_ultimo(lista):\n"
                  "    return (lista[0], lista[-2])\n"),
        "failing_test": "primeiro_ultimo([1, 2, 3]) == (1, 3)",
        "error": "AssertionError: primeiro_ultimo([1, 2, 3]) retornou (1, 2), esperado (1, 3)",
        "asserts": ["primeiro_ultimo([1, 2, 3]) == (1, 3)", "primeiro_ultimo([7]) == (7, 7)",
                    "primeiro_ultimo([4, 9]) == (4, 9)"],
    },
    {
        "id": "swe_contar_positivos", "name": "contar_positivos",
        "header": "def contar_positivos(numeros):\n",
        "buggy": ("def contar_positivos(numeros):\n"
                  "    conta = 0\n"
                  "    for n in numeros:\n"
                  "        if n > 0:\n"
                  "            conta += 1\n"
                  "        return conta\n"
                  "    return conta\n"),
        "failing_test": "contar_positivos([1, 2, 3]) == 3",
        "error": "AssertionError: contar_positivos([1, 2, 3]) retornou 1, esperado 3",
        "asserts": ["contar_positivos([1, 2, 3]) == 3", "contar_positivos([-1, 2, -3, 4]) == 2",
                    "contar_positivos([]) == 0", "contar_positivos([-5]) == 0"],
    },
    {
        "id": "swe_clamp", "name": "clamp",
        "header": "def clamp(valor, minimo, maximo):\n",
        "buggy": ("def clamp(valor, minimo, maximo):\n"
                  "    return max(maximo, min(minimo, valor))\n"),
        "failing_test": "clamp(5, 0, 10) == 5",
        "error": "AssertionError: clamp(5, 0, 10) retornou 10, esperado 5",
        "asserts": ["clamp(5, 0, 10) == 5", "clamp(-3, 0, 10) == 0",
                    "clamp(15, 0, 10) == 10", "clamp(0, 0, 10) == 0"],
    },
]


# ---------------------------------------------------------------------------
# Utilidades gerais (espelham cascade_c0/c3)
# ---------------------------------------------------------------------------

def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


def without_ipykernel_connection_args(argv: Iterable[str]) -> List[str]:
    """Remove '-f kernel-*.json' que o ipykernel injeta no Colab (espelha M0/C3)."""
    values = list(argv)
    filtered: List[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "-f" and index + 1 < len(values):
            name = Path(values[index + 1]).name
            if name.startswith("kernel-") and name.endswith(".json"):
                index += 2
                continue
        if value.startswith("-f="):
            name = Path(value[3:]).name
            if name.startswith("kernel-") and name.endswith(".json"):
                index += 1
                continue
        filtered.append(value)
        index += 1
    return filtered


def bootstrap_colab_secrets() -> None:
    """Espelha segredos do Colab (userdata) para env vars quando ausentes.

    Segredos NUNCA são gravados em arquivo — apenas ambiente do processo.
    RIFT_INGEST_TOKEN só é usado pelo publisher endurecido (HTTPS obrigatório
    + tamanho mínimo de 32 caracteres, ver publish_record).
    """
    names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "RIFT_INGEST_TOKEN", "RIFT_RESULTS_ENDPOINT")
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return
    for name in names:
        if os.environ.get(name, "").strip():
            continue
        try:
            value = str(userdata.get(name) or "").strip()
        except Exception:
            value = ""
        if value:
            os.environ[name] = value


def resolve_hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None


def ensure_hf_login(token: Optional[str] = None) -> Optional[str]:
    token = token or resolve_hf_token()
    if not token:
        return None
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=token, add_to_git_credential=False)
        print("[auth] HF_TOKEN aplicado.")
    except Exception as exc:
        print(f"[auth] AVISO: {exc}")
    return token


def resolve_device(s: str) -> "torch.device":
    s = (s or "auto").lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def schema_v2_fields(model_id: str, device_type: str, *,
                     backend: str = "transformers",
                     server_url: Optional[str] = None) -> Dict[str, Any]:
    """Campos obrigatórios do schema v2 (docs/C3_CONTRACTS_V1.md §3).

    No backend default (transformers) os campos são IDÊNTICOS aos anteriores;
    o backend llamacpp acrescenta backend/server_url ao comparison_context.
    """
    torch_v = (str(getattr(torch, "__version__", "unknown"))
               if torch is not None else "none")
    raw = f"{BENCHMARK_PROTOCOL}|{model_id}|{device_type}|{torch_v}"
    context: Dict[str, Any] = {
        "protocol": BENCHMARK_PROTOCOL,
        "device": device_type,
        "torch": torch_v,
        "transformers": _pkg_version("transformers"),
        "python": platform.python_version(),
    }
    if backend != "transformers":
        context["backend"] = backend
        context["server_url"] = server_url
    return {
        "schema_version": 2,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "comparison_group_id": "cmp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        "comparison_context": context,
        "implementation": {"kind": "REFERENCE_MEASURED", "native": False, "simulated": False},
    }


# ---------------------------------------------------------------------------
# Carregamento do modelo + geração greedy determinística
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: "torch.device", trust: bool, token: Optional[str]):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust)
    kwargs = dict(token=token, trust_remote_code=trust, low_cpu_mem_usage=True,
                  dtype=torch.float16 if device.type == "cuda" else torch.float32)
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if getattr(model, "hf_device_map", None) is None:
        model = model.to(device)
    model.eval()
    return model, tok


def generate_greedy(model, tok, prompt: str, max_new_tokens: int, device: "torch.device") -> str:
    """Continuação greedy determinística; retorna APENAS os tokens novos decodificados."""
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
        )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Backend llamacpp — geração greedy via HTTP no llama-server (contrato §11)
# ---------------------------------------------------------------------------

def _llamacpp_post(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    from urllib.request import Request, urlopen
    req = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                  headers={"Content-Type": "application/json",
                           "User-Agent": "capability-probe-battery/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def llamacpp_generate(server_url: str, prompt: str, max_new_tokens: int,
                      timeout_s: float = 300.0) -> str:
    """Geração greedy determinística via llama-server; retorna só o texto novo.

    Tenta a API nativa POST /completion e cai para /v1/completions
    (OpenAI-compatível) — ambas com parâmetros greedy (temperature 0, top_k 1).
    """
    base = server_url.rstrip("/")
    try:
        resp = _llamacpp_post(base + "/completion", {
            "prompt": prompt, "n_predict": max_new_tokens, "temperature": 0.0,
            "top_k": 1, "top_p": 1.0, "seed": 7, "stream": False,
            "cache_prompt": False,
        }, timeout_s)
        return str(resp.get("content") or "")
    except Exception as exc:
        print(f"[llamacpp] /completion indisponível ({exc}); tentando /v1/completions...")
    resp = _llamacpp_post(base + "/v1/completions", {
        "prompt": prompt, "max_tokens": max_new_tokens,
        "temperature": 0.0, "seed": 7,
    }, timeout_s)
    choices = resp.get("choices") or [{}]
    return str(choices[0].get("text") or "")


def check_llamacpp_server(server_url: str) -> None:
    """Falha cedo (exceção) se o llama-server não responder no /health."""
    from urllib.request import Request, urlopen
    if not str(server_url).lower().startswith(("http://", "https://")):
        raise RuntimeError(f"--server-url inválida: {server_url}")
    req = Request(server_url.rstrip("/") + "/health",
                  headers={"User-Agent": "capability-probe-battery/1.0"})
    with urlopen(req, timeout=30) as resp:
        status = int(getattr(resp, "status", 200))
        if status != 200:
            raise RuntimeError(f"llama-server /health retornou HTTP {status}")


# ---------------------------------------------------------------------------
# Sandbox de execução do CAP_CODING — sem import, sem IO, watchdog de 5s
# ---------------------------------------------------------------------------

def _blocked_import(name, *args, **kwargs):  # noqa: ANN001 - assinatura de __import__
    raise ImportError(f"import bloqueado no sandbox do CAP_CODING: {name}")


def make_safe_builtins() -> Dict[str, Any]:
    """Allowlist de builtins: sem open/eval/exec/getattr/globals e com import bloqueado."""
    allowed = {
        "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
        "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
        "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
        "max": max, "min": min, "ord": ord, "pow": pow, "range": range, "repr": repr,
        "reversed": reversed, "round": round, "set": set, "sorted": sorted, "str": str,
        "sum": sum, "tuple": tuple, "zip": zip,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "IndexError": IndexError, "KeyError": KeyError, "ZeroDivisionError": ZeroDivisionError,
        "StopIteration": StopIteration, "AssertionError": AssertionError,
        "True": True, "False": False, "None": None,
        "__import__": _blocked_import,
        "__name__": "cap_sandbox",
    }
    return allowed


def run_with_watchdog(fn, timeout_s: float) -> Dict[str, Any]:
    """Executa fn() em thread daemon com timeout (watchdog do sandbox).

    Uma thread Python não pode ser morta à força: em timeout a thread fica
    órfã (daemon) e a tarefa é marcada como falha — suficiente para o probe.
    """
    result: Dict[str, Any] = {"ok": False, "error": None, "timeout": False}

    def _target():
        try:
            fn()
            result["ok"] = True
        except BaseException as exc:  # noqa: BLE001 - sandbox reporta qualquer erro
            result["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        result["timeout"] = True
        result["error"] = f"timeout > {timeout_s:.0f}s (watchdog)"
    return result


def strip_markdown_fences(text: str) -> str:
    """Extrai o conteúdo do primeiro bloco ```...``` (ou remove crases soltas)."""
    match = re.search(r"```[a-zA-Z]*[ \t]*\r?\n(.*?)(?:```|\Z)", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.replace("```", "")


def _extract_full_function(code: str, name: str) -> Optional[str]:
    """Se o modelo reescreveu a função inteira (def na coluna 0), isola só ela."""
    match = re.search(rf"^def\s+{re.escape(name)}\s*\(", code, re.MULTILINE)
    if not match:
        return None
    lines = code[match.start():].splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            out.append(line)
            continue
        if line[0] in " \t":
            out.append(line)
            continue
        break
    return "\n".join(out) + "\n"


def build_candidate_source(task: Dict[str, Any], generated: str) -> str:
    """Header dado + corpo gerado -> fonte candidata para o sandbox.

    Dois modos de corpo: 'indent' (linhas já indentadas; para na primeira linha
    de coluna 0) e 'flat' (modelo colou o corpo na coluna 0; indenta tudo +4
    preservando indentação relativa).
    """
    code = strip_markdown_fences(generated)
    full = _extract_full_function(code, task["name"])
    if full is not None:
        return full
    body: List[str] = []
    mode: Optional[str] = None
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            if body:
                body.append("")
            continue
        indented = line[0] in " \t"
        if mode is None:
            mode = "indent" if indented else "flat"
        if mode == "indent" and not indented:
            break
        body.append(line if mode == "indent" else "    " + line)
    if not any(item.strip() for item in body):
        body = ["    pass"]
    return task["header"] + "\n".join(body) + "\n"


def evaluate_coding_task(task: Dict[str, Any], generated: str) -> Dict[str, Any]:
    """Compila e executa a fonte candidata no sandbox e roda os asserts embutidos."""
    source = build_candidate_source(task, generated)

    def _run():
        sandbox: Dict[str, Any] = {"__builtins__": make_safe_builtins()}
        exec(compile(source, "<cap_coding>", "exec"), sandbox)  # noqa: S102 - sandbox restrito
        fn = sandbox.get(task["name"])
        if not callable(fn):
            raise NameError(f"função {task['name']} não definida pelo código gerado")
        for expr in task["asserts"]:
            if not eval(compile(expr, "<cap_assert>", "eval"), sandbox):  # noqa: S307
                raise AssertionError(f"assert falhou: {expr}")

    outcome = run_with_watchdog(_run, CODING_TIMEOUT_S)
    ok = bool(outcome["ok"]) and not outcome["timeout"]
    detail = "todos os asserts passaram" if ok else str(outcome["error"] or "falha desconhecida")
    return {"id": task["id"], "ok": ok, "detail": detail[:300]}


# ---------------------------------------------------------------------------
# Parsing das saídas (múltipla escolha e JSON de function-calling)
# ---------------------------------------------------------------------------

def parse_mcq_letter(text: str) -> Optional[str]:
    """Primeira letra A-D da continuação (prioriza o padrão 'X)' das alternativas)."""
    upper = text.upper()
    match = re.search(r"([ABCD])\)", upper)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABCD])\b", upper)
    return match.group(1) if match else None


def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """Primeiro objeto JSON balanceado e parseável no texto (ou None)."""
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(text)):
            c = text[idx]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    try:
                        obj = json.loads(candidate)
                    except Exception:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    return None


# ---------------------------------------------------------------------------
# Runners por categoria
# ---------------------------------------------------------------------------

def mcq_prompt(question: Dict[str, Any]) -> str:
    opts = question["options"]
    return (
        "Responda a questão de múltipla escolha apenas com a letra correta.\n\n"
        f"Pergunta: {question['question']}\n"
        f"A) {opts['A']}\n"
        f"B) {opts['B']}\n"
        f"C) {opts['C']}\n"
        f"D) {opts['D']}\n"
        "Resposta:"
    )


def run_intelligence(generate, budget: int) -> Dict[str, Any]:
    tasks_out: List[Dict[str, Any]] = []
    hits = 0
    total = len(INTELLIGENCE_TASKS)
    for pos, question in enumerate(INTELLIGENCE_TASKS, start=1):
        text = generate(mcq_prompt(question), budget)
        predicted = parse_mcq_letter(text)
        ok = predicted == question["answer"]
        hits += int(ok)
        tasks_out.append({
            "id": question["id"],
            "ok": bool(ok),
            "detail": f"esperado={question['answer']} previsto={predicted or '?'}",
        })
        print(f"  [{pos:2d}/{total}] {question['id']}: {'OK' if ok else 'ERRO'} "
              f"(esperado {question['answer']}, previsto {predicted or '?'})")
    score = round(100.0 * hits / total, 2)
    return {"score": score, "tasks": tasks_out, "tasks_passed": hits,
            "tasks_total": total, "max_new_tokens": budget}


def coding_prompt(task: Dict[str, Any]) -> str:
    return (
        "# Complete a função Python abaixo. Escreva apenas o corpo da função.\n"
        + task["header"]
    )


def run_coding(generate, budget: int) -> Dict[str, Any]:
    tasks_out: List[Dict[str, Any]] = []
    passed = 0
    total = len(CODING_TASKS)
    for pos, task in enumerate(CODING_TASKS, start=1):
        text = generate(coding_prompt(task), budget)
        result = evaluate_coding_task(task, text)
        passed += int(result["ok"])
        tasks_out.append(result)
        print(f"  [{pos}/{total}] {task['id']}: {'OK' if result['ok'] else 'ERRO'} ({result['detail'][:80]})")
    score = round(100.0 * passed / total, 2)
    return {"score": score, "tasks": tasks_out, "tasks_passed": passed,
            "tasks_total": total, "max_new_tokens": budget}


def agentic_prompt(task: Dict[str, Any]) -> str:
    tool = task["tool"]
    schema = {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": {
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in tool["parameters"].items()},
            "required": list(tool["required"]),
        },
    }
    return (
        "Você tem acesso à ferramenta abaixo. Responda APENAS com uma chamada JSON "
        'no formato {"name": "<ferramenta>", "arguments": {...}}, sem texto extra.\n\n'
        f"Ferramenta: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Instrução: {task['instruction']}\n"
        "JSON:"
    )


def evaluate_agentic_task(task: Dict[str, Any], generated: str) -> Dict[str, Any]:
    """Pontuação: JSON parseável (50) + nome correto (25) + chaves obrigatórias (25)."""
    tool = task["tool"]
    points = 0
    details: List[str] = []
    obj = extract_first_json(generated)
    if obj is None:
        details.append("json=inválido")
    else:
        points += 50
        details.append("json=ok")
        if obj.get("name") == tool["name"]:
            points += 25
            details.append("nome=ok")
        else:
            details.append(f"nome={obj.get('name') or '?'}")
        arguments = obj.get("arguments")
        if isinstance(arguments, dict) and all(key in arguments for key in tool["required"]):
            points += 25
            details.append("args=ok")
        else:
            missing = [key for key in tool["required"]
                       if not (isinstance(arguments, dict) and key in arguments)]
            details.append("args_faltando=" + (",".join(missing) if missing else "estrutura"))
    return {"id": task["id"], "points": points, "detail": " ".join(details)[:300]}


def run_agentic(generate, budget: int) -> Dict[str, Any]:
    tasks_out: List[Dict[str, Any]] = []
    total = len(AGENTIC_TASKS)
    for pos, task in enumerate(AGENTIC_TASKS, start=1):
        text = generate(agentic_prompt(task), budget)
        result = evaluate_agentic_task(task, text)
        tasks_out.append(result)
        print(f"  [{pos}/{total}] {task['id']}: {result['points']} pts ({result['detail']})")
    score = round(sum(item["points"] for item in tasks_out) / total, 2)
    passed = sum(1 for item in tasks_out if item["points"] == 100)
    return {"score": score, "tasks": tasks_out, "tasks_passed": passed,
            "tasks_total": total, "max_new_tokens": budget}


# ---------------------------------------------------------------------------
# CAP_DEEPSEARCH_QA — QA multi-hop com match normalizado (§15)
# ---------------------------------------------------------------------------

def normalize_short_answer(text: str) -> str:
    """Normaliza para o match de QA: sem acentos, casefold, pontuação -> espaço."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = "".join(ch if ch.isalnum() else " " for ch in no_accents.casefold())
    return " ".join(cleaned.split())


def contains_normalized_answer(generated: str, answers: List[str]) -> bool:
    """Containment delimitado por palavra sobre o texto normalizado ('7' != '17')."""
    haystack = f" {normalize_short_answer(generated)} "
    for answer in answers:
        needle = normalize_short_answer(answer)
        if needle and f" {needle} " in haystack:
            return True
    return False


def deepsearch_prompt(task: Dict[str, Any]) -> str:
    lines = [
        "Leia as passagens abaixo e responda a pergunta combinando as informações "
        "das passagens. Responda de forma curta e direta (nome, número ou expressão).",
        "",
    ]
    for pos, passage in enumerate(task["passages"], start=1):
        lines.append(f"Passagem {pos}: {passage}")
    lines += ["", f"Pergunta: {task['question']}", "Resposta curta:"]
    return "\n".join(lines)


def run_deepsearch_qa(generate, budget: int) -> Dict[str, Any]:
    tasks_out: List[Dict[str, Any]] = []
    hits = 0
    total = len(DEEPSEARCH_TASKS)
    for pos, task in enumerate(DEEPSEARCH_TASKS, start=1):
        text = generate(deepsearch_prompt(task), budget)
        ok = contains_normalized_answer(text, task["answers"])
        hits += int(ok)
        snippet = " ".join(str(text).split())[:60]
        tasks_out.append({
            "id": task["id"],
            "ok": bool(ok),
            "detail": f"esperado={task['answers'][0]} saida={snippet or '?'}"[:300],
        })
        print(f"  [{pos}/{total}] {task['id']}: {'OK' if ok else 'ERRO'} "
              f"(esperado {task['answers'][0]})")
    score = round(100.0 * hits / total, 2)
    return {"score": score, "tasks": tasks_out, "tasks_passed": hits,
            "tasks_total": total, "max_new_tokens": budget}


# ---------------------------------------------------------------------------
# CAP_MCP_ATLAS — escolha de ferramenta MCP + chamada JSON (§15)
# ---------------------------------------------------------------------------

def mcp_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Renderiza um tool no formato de schema estilo MCP (inputSchema JSON Schema)."""
    return {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": {
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in tool["parameters"].items()},
            "required": list(tool["required"]),
        },
    }


def mcp_prompt(task: Dict[str, Any]) -> str:
    tools = [mcp_tool_schema(tool) for tool in task["tools"]]
    lines = [
        "Você é um agente com acesso às ferramentas MCP abaixo. Escolha a ferramenta "
        "correta e responda APENAS com uma chamada JSON no formato "
        '{"name": "<ferramenta>", "arguments": {...}}, sem texto extra.',
        "",
        "Ferramentas: " + json.dumps(tools, ensure_ascii=False),
    ]
    if task.get("step_result"):
        lines.append(task["step_result"])
    lines += [f"Instrução: {task['instruction']}", "JSON:"]
    return "\n".join(lines)


def _value_matches_type(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str) and bool(value.strip())
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    return value is not None


def _plausible_arguments(tool: Dict[str, Any], arguments: Any,
                         expect: Dict[str, List[str]]) -> "tuple[bool, str]":
    """Obrigatórios presentes + tipo correto + (quando há dica) valor plausível."""
    if not isinstance(arguments, dict):
        return False, "sem_objeto_arguments"
    problems: List[str] = []
    for key in tool["required"]:
        if key not in arguments:
            problems.append(f"faltando:{key}")
            continue
        value = arguments[key]
        if not _value_matches_type(value, tool["parameters"].get(key, "string")):
            problems.append(f"tipo:{key}")
            continue
        hints = expect.get(key)
        if hints:
            normalized = f" {normalize_short_answer(str(value))} "
            if not any(f" {hint} " in normalized for hint in hints):
                problems.append(f"valor:{key}")
    if problems:
        return False, ",".join(problems)[:120]
    return True, "ok"


def evaluate_mcp_task(task: Dict[str, Any], generated: str) -> Dict[str, Any]:
    """Pontuação §15: JSON parseável (40) + ferramenta correta (30) + args plausíveis (30)."""
    expected = next(tool for tool in task["tools"] if tool["name"] == task["expected_tool"])
    points = 0
    details: List[str] = []
    obj = extract_first_json(generated)
    if obj is None:
        details.append("json=inválido")
    else:
        points += 40
        details.append("json=ok")
        if obj.get("name") == expected["name"]:
            points += 30
            details.append("ferramenta=ok")
        else:
            details.append(f"ferramenta={obj.get('name') or '?'}")
        args_ok, args_detail = _plausible_arguments(
            expected, obj.get("arguments"), task.get("expect") or {})
        if args_ok:
            points += 30
            details.append("args=ok")
        else:
            details.append(f"args={args_detail}")
    return {"id": task["id"], "points": points, "detail": " ".join(details)[:300]}


def run_mcp_atlas(generate, budget: int) -> Dict[str, Any]:
    tasks_out: List[Dict[str, Any]] = []
    total = len(MCP_ATLAS_TASKS)
    for pos, task in enumerate(MCP_ATLAS_TASKS, start=1):
        text = generate(mcp_prompt(task), budget)
        result = evaluate_mcp_task(task, text)
        tasks_out.append(result)
        print(f"  [{pos}/{total}] {task['id']}: {result['points']} pts ({result['detail']})")
    score = round(sum(item["points"] for item in tasks_out) / total, 2)
    passed = sum(1 for item in tasks_out if item["points"] == 100)
    return {"score": score, "tasks": tasks_out, "tasks_passed": passed,
            "tasks_total": total, "max_new_tokens": budget}


# ---------------------------------------------------------------------------
# CAP_TAU3_BENCH — agente de atendimento com política declarada (§15)
# ---------------------------------------------------------------------------

def tau3_prompt(task: Dict[str, Any]) -> str:
    actions = [mcp_tool_schema(action) for action in task["actions"]]
    return (
        "Você é um agente de atendimento e DEVE seguir a política abaixo à risca "
        "(quando a política exigir, o correto é recusar ou escalar). Escolha a "
        "próxima ação correta e responda APENAS com uma chamada JSON no formato "
        '{"name": "<acao>", "arguments": {...}}, sem texto extra.\n\n'
        f"Política: {task['policy']}\n"
        f"Ações disponíveis: {json.dumps(actions, ensure_ascii=False)}\n"
        f"Solicitação do cliente: {task['request']}\n"
        "JSON:"
    )


def evaluate_tau3_task(task: Dict[str, Any], generated: str) -> Dict[str, Any]:
    """Pontuação §15: ação correta (60) + conformidade com a política (40)."""
    points = 0
    details: List[str] = []
    action_names = {action["name"] for action in task["actions"]}
    forbidden = set(task["forbidden_actions"])
    obj = extract_first_json(generated)
    name = obj.get("name") if isinstance(obj, dict) else None
    if obj is None:
        details.append("json=inválido (nenhuma ação identificável)")
    else:
        if name == task["expected_action"]:
            points += 60
            details.append("acao=ok")
        else:
            details.append(f"acao={name or '?'}")
        if isinstance(name, str) and name in action_names and name not in forbidden:
            points += 40
            details.append("politica=ok")
        elif name in forbidden:
            details.append("politica=violada")
        else:
            details.append("politica=indeterminada")
    return {"id": task["id"], "points": points, "detail": " ".join(details)[:300]}


def run_tau3_bench(generate, budget: int) -> Dict[str, Any]:
    tasks_out: List[Dict[str, Any]] = []
    total = len(TAU3_TASKS)
    for pos, task in enumerate(TAU3_TASKS, start=1):
        text = generate(tau3_prompt(task), budget)
        result = evaluate_tau3_task(task, text)
        tasks_out.append(result)
        print(f"  [{pos}/{total}] {task['id']}: {result['points']} pts ({result['detail']})")
    score = round(sum(item["points"] for item in tasks_out) / total, 2)
    passed = sum(1 for item in tasks_out if item["points"] == 100)
    return {"score": score, "tasks": tasks_out, "tasks_passed": passed,
            "tasks_total": total, "max_new_tokens": budget}


# ---------------------------------------------------------------------------
# CAP_SWE_BENCH — reparo de código no sandbox do CAP_CODING (§15)
# ---------------------------------------------------------------------------

def swe_prompt(task: Dict[str, Any]) -> str:
    return (
        "# A função Python abaixo tem um bug e o teste indicado falha.\n"
        "# Função com bug:\n"
        f"{task['buggy']}"
        f"# Teste que falha: assert {task['failing_test']}\n"
        f"# Erro observado: {task['error']}\n"
        f"# Escreva a versão corrigida COMPLETA da função {task['name']} "
        "(mesma assinatura), sem explicações:\n"
    )


def run_swe_bench(generate, budget: int) -> Dict[str, Any]:
    tasks_out: List[Dict[str, Any]] = []
    passed = 0
    total = len(SWE_TASKS)
    for pos, task in enumerate(SWE_TASKS, start=1):
        text = generate(swe_prompt(task), budget)
        # Reutiliza o sandbox/watchdog/extração do CAP_CODING (evaluate_coding_task).
        result = evaluate_coding_task(
            {"id": task["id"], "name": task["name"],
             "header": task["header"], "asserts": task["asserts"]},
            text)
        passed += int(result["ok"])
        tasks_out.append(result)
        print(f"  [{pos}/{total}] {task['id']}: {'OK' if result['ok'] else 'ERRO'} "
              f"({result['detail'][:80]})")
    score = round(100.0 * passed / total, 2)
    return {"score": score, "tasks": tasks_out, "tasks_passed": passed,
            "tasks_total": total, "max_new_tokens": budget}


# ---------------------------------------------------------------------------
# Recorder + publisher endurecido (upsert local + publish incremental)
# ---------------------------------------------------------------------------

def publish_record(rec: Dict[str, Any], endpoint: Optional[str] = None) -> None:
    """Publisher endurecido: HTTPS obrigatório + token >= 32 chars (contrato §5)."""
    endpoint = endpoint or os.environ.get("RIFT_RESULTS_ENDPOINT") or DEFAULT_ENDPOINT
    token = os.environ.get("RIFT_INGEST_TOKEN") or ""
    if len(token) < 32:
        print("[publish] skip (RIFT_INGEST_TOKEN ausente ou curto <32 chars)")
        return
    if not str(endpoint).lower().startswith("https://"):
        print(f"[publish] endpoint não-HTTPS bloqueado — skip: {endpoint}")
        return
    try:
        from urllib.request import Request, urlopen
        body = json.dumps({"records": [rec]}, ensure_ascii=False).encode("utf-8")
        req = Request(endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "capability-probe-battery/1.0",
        })
        with urlopen(req, timeout=60) as resp:
            print(f"[publish] HTTP {resp.status} battery={rec.get('battery_id')}")
    except Exception as exc:
        print(f"[publish] AVISO: {exc}")


class CapabilityRecorder:
    """Grava JSON local (upsert por battery_id) + CSV gêmeo + publish incremental."""

    CSV_FIELDS = [
        "timestamp_utc", "run_id", "technology", "model_id", "battery_id", "status",
        "capability_score", "tasks_passed", "tasks_total", "measurement_scope",
    ]

    def __init__(self, out_dir: Path, *, model_id: str, run_id: str,
                 schema_fields: Dict[str, Any], publish_on: bool,
                 endpoint: Optional[str] = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / "cap_test_batteries.json"
        self.csv_path = out_dir / "cap_test_batteries.csv"
        self.model_id = model_id
        self.run_id = run_id
        self.schema_fields = schema_fields
        self.publish_on = publish_on
        self.endpoint = endpoint
        self.records: List[Dict[str, Any]] = []
        if self.json_path.is_file():
            try:
                existing = json.loads(self.json_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    self.records = existing
            except Exception:
                self.records = []

    def emit(self, battery_id: str, status: str, *, capability: Dict[str, Any],
             scope: str, notes: str, error: Optional[str] = None) -> Dict[str, Any]:
        global EMITTED_RECORDS
        metrics: Dict[str, Any] = {"capability": capability}
        if error:
            metrics["error"] = error[:800]
        rec = {
            "timestamp_utc": utc(),
            "run_id": self.run_id,
            "technology": "CAP",
            "model_id": self.model_id,
            "battery_id": battery_id,
            "status": status,
            **self.schema_fields,
            "comparison_role": None,
            "eligible_for_primary_ranking": False,
            "baseline_tok_s": None,
            "candidate_tok_s": None,
            "baseline_ram_bytes": None,
            "candidate_ram_bytes": None,
            "baseline_disk_bytes": None,
            "candidate_disk_bytes": None,
            "measurement_scope": scope,
            "quality": None,
            "metrics": metrics,
            "notes": notes[:1200],
        }
        # upsert por battery_id no arquivo consolidado local
        self.records = [item for item in self.records if item.get("battery_id") != battery_id]
        self.records.append(rec)
        self.records.sort(key=lambda item: str(item.get("battery_id")))
        self.json_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        single = self.batteries_dir / f"{self.run_id}__{battery_id}.json"
        single.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._write_csv()
        EMITTED_RECORDS += 1
        print(f"[BATTERY] {battery_id}: gravada -> {single}")
        if self.publish_on:
            publish_record(rec, self.endpoint)
        return rec

    def _write_csv(self) -> None:
        """CSV gêmeo reescrito a partir do JSON (upsert determinístico)."""
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for item in self.records:
                cap = (item.get("metrics") or {}).get("capability") or {}
                writer.writerow({
                    "timestamp_utc": item.get("timestamp_utc"),
                    "run_id": item.get("run_id"),
                    "technology": item.get("technology"),
                    "model_id": item.get("model_id"),
                    "battery_id": item.get("battery_id"),
                    "status": item.get("status"),
                    "capability_score": cap.get("score"),
                    "tasks_passed": cap.get("tasks_passed"),
                    "tasks_total": cap.get("tasks_total"),
                    "measurement_scope": item.get("measurement_scope"),
                })


# ---------------------------------------------------------------------------
# Montagem dos registros e tabela final
# ---------------------------------------------------------------------------

# Ordem FIXA de categorias (§15): Intelligence, Coding, Agentic, DeepSearch QA,
# MCP-Atlas, τ³-Bench, SWE-Bench — usada na execução, no FAIL de infraestrutura
# e na tabela-resumo do console.
CATEGORY_ORDER = [
    "CAP_INTELLIGENCE",
    "CAP_CODING",
    "CAP_AGENTIC",
    "CAP_DEEPSEARCH_QA",
    "CAP_MCP_ATLAS",
    "CAP_TAU3_BENCH",
    "CAP_SWE_BENCH",
]

CATEGORY_LABELS = {
    "CAP_INTELLIGENCE": "Inteligência",
    "CAP_CODING": "Código",
    "CAP_AGENTIC": "Agêntico",
    "CAP_DEEPSEARCH_QA": "DeepSearch QA",
    "CAP_MCP_ATLAS": "MCP-Atlas",
    "CAP_TAU3_BENCH": "τ³-Bench",
    "CAP_SWE_BENCH": "SWE-Bench",
}

# Nomes de exibição para os dashboards (§15) — gravados em
# metrics.capability.display_name APENAS nas categorias novas (as 3 originais
# permanecem byte-compatíveis, sem campos adicionais).
CATEGORY_DISPLAY_NAMES = {
    "CAP_DEEPSEARCH_QA": "DeepSearch QA",
    "CAP_MCP_ATLAS": "MCP-Atlas",
    "CAP_TAU3_BENCH": "τ³-Bench",
    "CAP_SWE_BENCH": "SWE-Bench",
}

# Rótulo de honestidade por categoria nova (§15): "probe leve inspirado em
# <benchmark oficial> — NÃO é o benchmark oficial completo". As 3 categorias
# originais mantêm o HONEST_LABEL histórico (byte-compatibilidade).
CATEGORY_HONEST_LABELS = {
    "CAP_DEEPSEARCH_QA": ("probe leve inspirado em DeepSearch QA — NÃO é o "
                          "benchmark oficial completo"),
    "CAP_MCP_ATLAS": ("probe leve inspirado em MCP Atlas — NÃO é o "
                      "benchmark oficial completo"),
    "CAP_TAU3_BENCH": ("probe leve inspirado em τ³-Bench — NÃO é o "
                       "benchmark oficial completo"),
    "CAP_SWE_BENCH": ("probe leve inspirado em SWE-Bench — NÃO é o "
                      "benchmark oficial completo"),
}


def honest_label_for(battery_id: str) -> str:
    return CATEGORY_HONEST_LABELS.get(battery_id, HONEST_LABEL)


def category_scope(battery_id: str) -> str:
    if battery_id == "CAP_INTELLIGENCE":
        base = (f"{len(INTELLIGENCE_TASKS)} questões de múltipla escolha embutidas "
                "(conhecimentos gerais/lógica/matemática, PT-BR/EN); geração greedy; "
                "score = % de acertos")
    elif battery_id == "CAP_CODING":
        base = (f"{len(CODING_TASKS)} completações de função Python com asserts embutidos, "
                "executadas em sandbox restrito (sem import/IO, watchdog "
                f"{CODING_TIMEOUT_S:.0f}s); score = % de tarefas com todos os asserts passando")
    elif battery_id == "CAP_DEEPSEARCH_QA":
        base = (f"{len(DEEPSEARCH_TASKS)} tarefas de QA multi-hop com mini-corpus embutido "
                "(2-3 passagens fictícias PT-BR/EN por prompt; entidades inventadas para "
                "evitar memorização); a resposta exige combinar fatos entre passagens; "
                "score = % de acertos por match normalizado (casefold, sem acentos/pontuação)")
    elif battery_id == "CAP_MCP_ATLAS":
        base = (f"{len(MCP_ATLAS_TASKS)} tarefas de uso de ferramentas estilo MCP (3-4 tool "
                "schemas por prompt, incluindo 2 cenários multi-passo com o resultado do "
                "passo 1 embutido); pontos: JSON parseável (40) + ferramenta correta (30) + "
                "argumentos obrigatórios com valores plausíveis (30); score = média dos pontos")
    elif battery_id == "CAP_TAU3_BENCH":
        base = (f"{len(TAU3_TASKS)} tarefas de agente de atendimento com política declarada "
                "(domínios bancário/aéreo; em alguns casos o correto é recusar ou escalar); "
                "pontos: ação correta (60) + conformidade com a política (40); mesma "
                "convenção de chamada JSON; score = média dos pontos")
    elif battery_id == "CAP_SWE_BENCH":
        base = (f"{len(SWE_TASKS)} reparos de código (função Python com bug + teste falhando "
                "+ erro no prompt); execução no mesmo sandbox restrito do CAP_CODING (sem "
                f"import/IO, watchdog {CODING_TIMEOUT_S:.0f}s); score = % de tarefas com "
                "toda a suíte de asserts embutida passando")
    else:
        base = (f"{len(AGENTIC_TASKS)} tarefas de function-calling em JSON contra schema "
                "embutido; pontos: JSON parseável (50) + nome da ferramenta (25) + "
                "chaves obrigatórias (25); score = média dos pontos")
    return (f"{BENCHMARK_PROTOCOL} {battery_id}: {base}; {honest_label_for(battery_id)}; "
            "não mede tok/s, RAM nem disco (campos de topo nulos); "
            "score é medida, não gate (status PASS = suíte completou)")


def category_notes(battery_id: str, model_id: str, device_type: str, budget: int) -> str:
    return (f"Probe de capacidade {CATEGORY_LABELS[battery_id]} ({battery_id}); "
            f"modelo={model_id}; device={device_type}; geração greedy determinística "
            f"(max_new_tokens={budget}); {honest_label_for(battery_id)}.")


def print_final_table(rows: List[Dict[str, Any]]) -> None:
    """Tabela-resumo final em PT-BR: as 7 categorias, score e tarefas aprovadas."""
    print("\n===== CAPABILITY_PROBE_V1 — resumo =====")
    print(f"{'Categoria':<34} {'Status':<8} {'Score':>7} {'Tarefas OK':>12}")
    print("-" * 66)
    scores: List[float] = []
    for row in rows:
        label = f"{CATEGORY_LABELS[row['battery_id']]} ({row['battery_id']})"
        if row["status"] == "PASS" and row.get("result"):
            result = row["result"]
            score_txt = f"{result['score']:7.1f}"
            tasks_txt = f"{result['tasks_passed']}/{result['tasks_total']}"
            scores.append(float(result["score"]))
        else:
            score_txt = f"{'—':>7}"
            tasks_txt = "—"
        print(f"{label:<34} {row['status']:<8} {score_txt} {tasks_txt:>12}")
    print("-" * 66)
    if scores:
        print(f"{'Média geral':<34} {'':<8} {sum(scores) / len(scores):7.1f}")
    else:
        print("Nenhuma categoria concluída — verifique os registros FAIL.")
    print(f"({HONEST_LABEL}; categorias novas: probes leves inspirados nos "
          "benchmarks citados — NÃO são os benchmarks oficiais completos)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        description="Bateria de capacidades CAPABILITY_PROBE_V1 (CAP_INTELLIGENCE / "
                    "CAP_CODING / CAP_AGENTIC / CAP_DEEPSEARCH_QA / CAP_MCP_ATLAS / "
                    "CAP_TAU3_BENCH / CAP_SWE_BENCH) — probes leves embutidos, "
                    "estilo OpenRouter compare (docs/C3_CONTRACTS_V1.md §9/§15).")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                   help="model_id do Hugging Face avaliado (baseline, sem otimização)")
    p.add_argument("--backend", default="transformers",
                   choices=["transformers", "llamacpp"],
                   help="backend de geração (default: transformers, sem mudança de "
                        "comportamento; llamacpp usa HTTP no llama-server)")
    p.add_argument("--server-url", default="http://127.0.0.1:8090",
                   help="URL do llama-server para --backend llamacpp "
                        "(default: %(default)s)")
    p.add_argument("--model-id-label", default=None,
                   help="override do model_id gravado nos registros (default: valor "
                        'de --model; ex.: "unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL")')
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="dispositivo de inferência (default: auto; ignorado no "
                        "backend llamacpp)")
    p.add_argument("--out", default="cap_test_output",
                   help="diretório de saída dos artefatos locais")
    p.add_argument("--publish", default="on", choices=["on", "off"],
                   help="publica cada registro no endpoint de resultados (default: on)")
    p.add_argument("--max-new-tokens", type=int, default=24,
                   help="orçamento-base de geração; cada categoria deriva o seu "
                        "(múltipla escolha <= 6; QA multi-hop <= 24; agêntico/MCP/"
                        "política >= 48; código >= 96; reparo de código >= 128)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="permite código remoto do repositório do modelo (AVISO de segurança)")
    p.add_argument("--results-endpoint", default=None,
                   help="override do endpoint HTTPS de publicação")
    args = p.parse_args(without_ipykernel_connection_args(values))

    bootstrap_colab_secrets()
    if args.trust_remote_code:
        print("[segurança] AVISO: --trust-remote-code executa código do repositório do "
              "modelo no Hugging Face. Use apenas com modelos de fonte confiável.")

    if args.backend == "transformers":
        if _MISSING_DEPS:
            raise SystemExit(MISSING_DEPS_MESSAGE)
        torch.manual_seed(0)
        device = resolve_device(args.device)
        device_type = device.type
    else:
        device = None
        device_type = "llamacpp"
    source_model = args.model.strip().replace("https://huggingface.co/", "").strip("/")
    model_id = (args.model_id_label.strip() if args.model_id_label else source_model)
    out_dir = Path(args.out)
    run_id = make_run_id()
    recorder = CapabilityRecorder(
        out_dir,
        model_id=model_id,
        run_id=run_id,
        schema_fields=schema_v2_fields(
            model_id, device_type, backend=args.backend,
            server_url=args.server_url if args.backend == "llamacpp" else None),
        publish_on=args.publish != "off",
        endpoint=args.results_endpoint,
    )

    # Orçamentos de geração por categoria, derivados de --max-new-tokens:
    # múltipla escolha só precisa da letra; QA multi-hop precisa de uma resposta
    # curta (~24); chamadas JSON (MCP/política) de ~48; código de um corpo
    # inteiro; reparo de código de uma função completa.
    base = max(1, int(args.max_new_tokens))
    budgets = {
        "CAP_INTELLIGENCE": max(2, min(base, 6)),
        "CAP_CODING": max(base, 96),
        "CAP_AGENTIC": max(base, 48),
        "CAP_DEEPSEARCH_QA": max(8, min(base, 24)),
        "CAP_MCP_ATLAS": max(base, 48),
        "CAP_TAU3_BENCH": max(base, 48),
        "CAP_SWE_BENCH": max(base, 128),
    }

    print(f"[CAP] model={model_id} backend={args.backend} device={device_type} "
          f"publish={args.publish} orçamentos={budgets}")

    table_rows: List[Dict[str, Any]] = []

    def _fail_capability(battery_id: str) -> Dict[str, Any]:
        # Payload de FAIL por categoria; as 3 originais seguem byte-compatíveis
        # (sem display_name), as novas ganham display_name + honest_label do §15.
        capability: Dict[str, Any] = {
            "category": battery_id, "score": None, "tasks": [],
            "tasks_passed": 0, "tasks_total": 0,
            "honest_label": honest_label_for(battery_id),
        }
        if battery_id in CATEGORY_DISPLAY_NAMES:
            capability["display_name"] = CATEGORY_DISPLAY_NAMES[battery_id]
        return capability

    def _fail_all_categories(reason: str, error_text: str) -> int:
        # Erro de infraestrutura: as 7 categorias são registradas como FAIL (exit 0).
        for battery_id in CATEGORY_ORDER:
            recorder.emit(
                battery_id, "FAIL",
                capability=_fail_capability(battery_id),
                scope=category_scope(battery_id),
                notes=(f"FAIL de infraestrutura ({reason}); modelo={model_id}; "
                       f"device={device_type}; {honest_label_for(battery_id)}."),
                error=error_text[:800],
            )
            table_rows.append({"battery_id": battery_id, "status": "FAIL", "result": None})
        print_final_table(table_rows)
        return 0

    if args.backend == "transformers":
        hf_token = ensure_hf_login()
        try:
            model, tok = load_model(source_model, device, args.trust_remote_code, hf_token)
        except Exception as exc:
            traceback.print_exc()
            return _fail_all_categories(
                "modelo não carregou", f"Falha ao carregar o modelo: {exc}")

        def generate(prompt: str, budget: int) -> str:
            return generate_greedy(model, tok, prompt, budget, device)
    else:
        try:
            check_llamacpp_server(args.server_url)
        except Exception as exc:
            traceback.print_exc()
            return _fail_all_categories(
                "llama-server inacessível",
                f"llama-server inacessível em {args.server_url}: {exc}")

        def generate(prompt: str, budget: int) -> str:
            return llamacpp_generate(args.server_url, prompt, budget)

    # Ordem fixa do §15 (CATEGORY_ORDER): Intelligence, Coding, Agentic,
    # DeepSearch QA, MCP-Atlas, τ³-Bench, SWE-Bench.
    runners = {
        "CAP_INTELLIGENCE": run_intelligence,
        "CAP_CODING": run_coding,
        "CAP_AGENTIC": run_agentic,
        "CAP_DEEPSEARCH_QA": run_deepsearch_qa,
        "CAP_MCP_ATLAS": run_mcp_atlas,
        "CAP_TAU3_BENCH": run_tau3_bench,
        "CAP_SWE_BENCH": run_swe_bench,
    }
    categories = [(battery_id, runners[battery_id]) for battery_id in CATEGORY_ORDER]
    for battery_id, runner in categories:
        budget = budgets[battery_id]
        print(f"\n[CAP] Executando {battery_id} ({CATEGORY_LABELS[battery_id]}, "
              f"max_new_tokens={budget})...")
        try:
            result = runner(generate, budget)
        except Exception as exc:
            # Erro de infraestrutura da categoria: FAIL; o score não é gate.
            traceback.print_exc()
            recorder.emit(
                battery_id, "FAIL",
                capability=_fail_capability(battery_id),
                scope=category_scope(battery_id),
                notes=(f"FAIL de infraestrutura durante a suíte; modelo={model_id}; "
                       f"device={device_type}; {honest_label_for(battery_id)}."),
                error=str(exc)[:800],
            )
            table_rows.append({"battery_id": battery_id, "status": "FAIL", "result": None})
            continue
        capability = {
            "category": battery_id,
            "score": result["score"],
            "tasks": result["tasks"],
            "tasks_passed": result["tasks_passed"],
            "tasks_total": result["tasks_total"],
            "max_new_tokens": result["max_new_tokens"],
            "honest_label": honest_label_for(battery_id),
        }
        if battery_id in CATEGORY_DISPLAY_NAMES:
            # §15: nome de exibição da categoria para os dashboards (só nas novas;
            # as 3 originais permanecem byte-compatíveis).
            capability["display_name"] = CATEGORY_DISPLAY_NAMES[battery_id]
        recorder.emit(
            battery_id, "PASS",
            capability=capability,
            scope=category_scope(battery_id),
            notes=category_notes(battery_id, model_id, device_type, budget),
        )
        table_rows.append({"battery_id": battery_id, "status": "PASS", "result": result})

    print_final_table(table_rows)
    print(f"\n[CAP] Artefatos locais: {recorder.json_path} | {recorder.csv_path} | "
          f"{recorder.batteries_dir}")
    return 0


if __name__ == "__main__":
    try:
        _rc = main() or 0
    except SystemExit as _exc:
        _rc = int(_exc.code) if isinstance(_exc.code, int) else 1
    except Exception:
        traceback.print_exc()
        # Não-zero apenas em crash ANTES de qualquer registro gravado.
        _rc = 0 if EMITTED_RECORDS > 0 else 1
    raise SystemExit(_rc)
