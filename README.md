Versão  - https://safety-chat.streamlit.app/ - app_safety_chat.py  originario do app_chat_novo_correto.py

## Atualizacao dos artefatos de analytics (MVP)

O Safety Chat consome artefatos estaticos em `data/analytics`. As fontes monitoradas
ficam em `data/xlsx` e `data/docs`, e o estado esperado dessas fontes fica registrado
em `data/analytics/manifest.json`.

### Rotina recomendada: atualizacao local segura

Para atualizar os artefatos sem digitar comandos, execute na raiz do repositorio:

```text
Atualizar_Analytics_SafetyChat.bat
```

Essa rotina chama `tools/update_analytics_local.py` e faz o processo em etapas:

1. verifica se houve mudanca monitorada;
2. se apenas `data/docs` mudou, valida os artefatos atuais e atualiza somente o manifesto;
3. se `data/xlsx` mudou, gera os novos artefatos em `.analytics_build_tmp`;
4. valida os arquivos gerados antes de tocar em `data/analytics`;
5. cria backup dos artefatos atuais em `.analytics_backups`;
6. substitui os arquivos de `data/analytics` somente depois da validacao;
7. valida novamente os arquivos finais e confere se o manifesto ficou sincronizado.

Se algum erro ocorrer durante a substituicao, a rotina tenta restaurar o backup. Ao final,
revise as alteracoes no GitHub Desktop, faca commit e push.

Tambem e possivel executar a rotina pelo Python:

```powershell
python tools/update_analytics_local.py
```

Opcoes uteis:

```powershell
python tools/update_analytics_local.py --force
python tools/update_analytics_local.py --dry-run
python tools/update_analytics_local.py --families sphera
python tools/update_analytics_local.py --families ws,precursors,cp
```

Por padrao, `--families auto` tenta reconstruir apenas a familia associada a planilha
alterada. Se detectar uma planilha nova ou sem mapeamento conhecido, reconstrui todas.

### Comandos manuais do gerador

Verificar se as fontes mudaram:

```powershell
python tools/build_analytics.py --check
```

Atualizar somente o manifesto, sem recriar embeddings:

```powershell
python tools/build_analytics.py --manifest-only
```

Validar os artefatos atuais:

```powershell
python tools/build_analytics.py --validate
```

Reconstruir os artefatos usados pelo app atual:

```powershell
python tools/build_analytics.py --build
```

Tambem e possivel reconstruir apenas uma familia:

```powershell
python tools/build_analytics.py --build --families sphera
python tools/build_analytics.py --build --families ws,precursors,cp
```

Depois de uma reconstrucao real, valide o app localmente, faca commit dos artefatos
alterados em `data/analytics` e envie a branch para o GitHub. O Streamlit deve apenas
consumir os artefatos versionados; a geracao pesada nao deve rodar no runtime do app.

1) Estrutura recomendada de pastas
eso-chat/
├─ app_chat.py                 # app principal (Streamlit)
├─ requirements.txt            # deps leves (streamlit, requests, numpy, pandas, pypdf, python-docx)
├─ README.md
├─ .gitignore
├─ .gitattributes              # opcional (Git LFS p/ PDFs/DOCX/NPZ)
└─ data/
   └─ analytics/               # opcional (csv/npz estáticos, se quiser versionar)

2) Pré-requisitos
Conta no Streamlit Cloud (grátis).
Conta no Ollama Cloud: https://ollama.com
Em Settings → API Keys, crie uma API key.
Repositório no GitHub com os arquivos acima.

3) Configurar secrets no Streamlit Cloud

No dashboard do Streamlit Cloud:

Deploy app a partir do GitHub (seu repo → app_chat.py).

Em App → Settings → Secrets, cole:

# Obrigatório
OLLAMA_API_KEY = "coloque_sua_api_key_aqui"

# Opcionais (deixe assim se usar o Ollama Cloud)

OLLAMA_HOST  = "https://api.ollama.com" 

OLLAMA_MODEL = "llama3.1"    # ou outro modelo disponível no Ollama Cloud

Salve os secrets.

4) requirements.txt (exemplo)
streamlit==1.36.0
requests
numpy
pandas
pypdf
python-docx

Não adicionamos torch ou sentence-transformers.

5) .gitignore (sugestão)
# Python / Streamlit
__pycache__/
*.pyc
.venv/
env/
.conda/
.ipynb_checkpoints/
.streamlit/secrets.toml

# Artefatos locais
data/storage/
data/tmp/
tmp/
*.log

# Se decidir gerar em runtime:
*.npz
*.npy

# SO / IDE
.DS_Store
Thumbs.db
.vscode/
.idea/

6) (Opcional) Git LFS para arquivos grandes

Se quiser versionar PDFs/DOCX/NPZ:
Instale Git LFS e rode:

git lfs install
git lfs track "*.pdf"
git lfs track "*.docx"
git lfs track "*.npz"


Isto criará/atualizará .gitattributes com algo como:

*.pdf  filter=lfs diff=lfs merge=lfs -text
*.docx filter=lfs diff=lfs merge=lfs -text
*.npz  filter=lfs diff=lfs merge=lfs -text
Commit e push normalmente.

7) Como funciona o app
Upload: PDF, DOCX, XLSX, CSV, TXT/MD.
O app faz chunking do texto e gera embeddings no Ollama Cloud (/api/embeddings, modelo nomic-embed-text).
Na pergunta do usuário, calcula similaridade cosseno vs. índice e envia contexto relevante ao modelo de chat (/api/chat).
Ajustes no sidebar:
Tamanho/overlap de chunk
Top-K de contexto
Limiar de similaridade

8) Teste rápido da API (local, opcional)
cURL
curl -s https://api.ollama.com/api/chat \
  -H "Authorization: Bearer $OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.1","messages":[{"role":"user","content":"diga ok"}],"stream":false}'

Python
import os, requests
host = os.getenv("OLLAMA_HOST","https://api.ollama.com")
key  = os.getenv("OLLAMA_API_KEY")  # export OLLAMA_API_KEY=...

r = requests.post(
    f"{host}/api/chat",
    headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"},
    json={"model":"llama3.1", "messages":[{"role":"user","content":"diga ok"}], "stream":False},
    timeout=60
)
print(r.status_code, r.json())

9) Deploy
Faça push para o GitHub.
No Streamlit Cloud, crie o app apontando para app_chat.py.
Configure os Secrets (seção 3).
Clique em Deploy.

10) Solução de problemas
401 Unauthorized: verifique OLLAMA_API_KEY nos Secrets.
Time-out embeddings: uploads muito grandes? Faça em lotes menores; ajuste chunk size.
PDF vazio: alguns PDFs “scanneados” não têm texto extraível. Use OCR antes.
Erros de pacote: confira requirements.txt e reimplante.

11) Próximos passos (sugestões)
Persistir índice em data/analytics (CSV/NPZ) e permitir download/upload do índice.
Implementar fonte citada (arquivo e chunk_id) no rodapé da resposta.
Adicionar controles avançados (temperature, max tokens, penalidades).
Incluir métricas básicas (latência, tokens estimados).

12) Licença
Defina a licença que preferir (por exemplo, MIT) no repositório.
