# Classificação de Reservas de Hotel

API desenvolvida com **FastAPI** para realizar inferências com um modelo de Machine Learning treinado para classificar reservas de hotel.

---

## Sumário
1. [Processo de desenvolvimento](#processo-de-desenvolvimento)
2. [Execução do Projeto em Servidor Local](#execucao-do-projeto-em-servidor-local)
3. [Execução com Docker](#execucao-com-docker)
4. [Deploy na AWS](#deploy-na-aws)
5. [Resultados](#resultados)

---

## Processo de Desenvolvimento

### Estrutura de Pastas

```bash
/app
   ├── config/
       ├── .env           # Variáveis de ambiente (não commitado no Git)
       ├── .env.example   # Exemplo de variáveis de ambiente
   ├── scripts/
       ├── datasets/
           ├── Hotel Reservations.csv   # Dataset original das reservas
       ├── hotel_test_xgboost.csv       # XGBoost: Parte de teste 
       ├── hotel_train_xgboost.csv      # XGBoost: Parte de treinamento
       ├── hotel-treinamento.ipynb      # Notebook para treinamento do modelo
       ├── modelo_xgboost_treinado.pkl  # Modelo treinado e com 89% de acurácia
   ├── main.py             # Código principal da API FastAPI
   ├── Dockerfile          # Definição do ambiente Docker
   ├── requirements.txt    # Dependências do projeto
.gitignore                 # Arquivos ignorados no controle de versão
README.md                  # Documentação do projeto
```

> *Dica*: Após configurar o `.env`, o arquivo `.env.example` pode ser removido, pois é apenas um exemplo.

### Criação do Modelo

1. **Preparação dos Dados**:
   - **Dataset Original**: `Hotel Reservations.csv`
   - **Criação da Coluna de Rótulo**: `label_avg_price_per_room` foi criada com base na coluna `avg_price_per_room` para classificar os dados em três faixas de preço.

2. **Treinamento do Modelo**:
   - **Modelo Utilizado**: XGBoost (`xgb.XGBClassifier`)
   - **Preprocessamento**:
     - Excluída a coluna `avg_price_per_room`.
     - Criada a nova coluna `label_avg_price_per_room`.
   - **Treinamento e Avaliação**:
     - Treinado o modelo com os dados processados.
     - Avaliado o modelo com uma acurácia de 89%.

3. **Salvamento do Modelo**:
   - O modelo treinado foi salvo localmente como `modelo_xgboost_treinado.pkl`.

### Desenvolvimento da API

1. **Configuração do Ambiente**:
   - **Dependências**: Utilizado `FastAPI` para o desenvolvimento da API.
   - **Estrutura**:
     - **main.py**: Código principal da API FastAPI, que carrega o modelo treinado e expõe um endpoint `/api/v1/inference`.
     - **Dockerfile**: Configuração para criar uma imagem Docker que contenha o ambiente necessário para a execução da API.

2. **Endpoint de Inferência**:
   - **Rota**: `/api/v1/inference`
   - **Método**: POST
   - **Formato do JSON de Entrada**:
     ```json
     {
         "no_of_adults": 3,
         "no_of_children": 3,
         "type_of_meal_plan": "example",
         "room_type_reserved": 3,
         "arrival_year": 2020,
         "market_segment_type": 1
     }
     ```
   - **Resposta**:
     ```json
     {
       "result": 1
     }
     ```

3. **Processamento das Entradas**:
   - **Função `process_input`**: Prepara as entradas para o modelo, gerando um vetor de características com o formato esperado pelo modelo.

---

## Execução do Projeto em Servidor Local

### Pré-requisitos
- **Python 3.10**
   - Windows: Faça o download no [site oficial do Python](https://www.python.org/downloads/)
   - Linux:
     ```bash
     sudo apt install python3.10
     ```
- **Docker**
    - Windows: Faça o download no [site oficial do Docker](https://www.docker.com/products/docker-desktop/)
    - Linux:
     ```bash
     sudo apt install docker-ce
     docker --version
     ```

- **Postman**: Faça o download no [site oficial do Postman](https://www.postman.com/downloads/)

---

### Clonando o Projeto

1. Clone este repositório na sua máquina local:
   ```bash
   git clone https://github.com/hellenilda/HotelReservations.git
   ```

2. Navegue até o diretório do projeto:
   ```bash
   cd HotelReservations
   ```

### Criação e Configuração do Virtualenv

1. Instale o `virtualenv` e o `virtualenvwrapper`:
   ```bash
   pip install virtualenv virtualenvwrapper
   ```

2. Crie o ambiente virtual:
   ```bash
   mkvirtualenv nome-do-virtualenv -p python3.10
   ```

3. Ative o ambiente virtual (caso não tenha sido ativado após a criação):
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **Linux/macOS**:
     ```bash
     source /home/usuario/.virtualenvs/nome-do-virtualenv/bin/activate
     ```

4. Instale as dependências do projeto:
   ```bash
   pip install -r requirements.txt
   ```

5. Acesse o diretório `app/` e execute o uvicorn:
   ```bash
   cd app/
   uvicorn main:app --reload
   ```

6. Abra o Postman e execute as inferências em uma requisição POST através da URL http://127.0.0.1:8000/api/v1/inference.
   Exemplo de inferência:
   ```json
   {
       "no_of_adults": 3,
       "no_of_children": 3,
       "type_of_meal_plan": "example",
       "room_type_reserved": 3,
       "arrival_year": 2020,
       "market_segment_type": 1
   }
   ```

   Resultado esperado:
   ```json
   {
     "result": 1
   }
   ```

---

<!-- ## Deploy na AWS

### [Informações sobre o processo de deploy na AWS] -->

### Execução com Docker

Para garantir que a aplicação FastAPI seja executada de forma consistente em diferentes ambientes, você pode containerizá-la utilizando Docker. A seguir estão os passos e a configuração necessária para isso.
> **OBS.**: Não precisa realizar o processo de execução em servidor local se for conteinerizar a aplicação com Docker.

### Dockerfile

O `Dockerfile` foi configurado para criar uma imagem Docker que contém a aplicação FastAPI e todas as suas dependências. Aqui está o conteúdo do `Dockerfile`:

```Dockerfile
# Usar uma imagem base com Python 3.10
FROM python:3.10-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos do projeto para o diretório de trabalho
COPY . /app

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta onde a aplicação será executada
EXPOSE 8000

# Comando para iniciar a aplicação com uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Construir a Imagem Docker

Para construir a imagem Docker, execute o seguinte comando no diretório onde está localizado o `Dockerfile`:

```bash
docker build -t hotel-reservations-api .
```

### Executar o Contêiner

Depois de construir a imagem, você pode executar o contêiner com o seguinte comando:

```bash
docker run -p 8000:8000 hotel-reservations-api
```

Este comando mapeia a porta 8000 do contêiner para a porta 8000 da sua máquina local. A aplicação estará acessível em `http://127.0.0.1:8000`.

### Verificar a Aplicação

Para verificar se a aplicação está rodando corretamente, abra um navegador e acesse `http://127.0.0.1:8000/docs`. Isso deve exibir a interface do Swagger gerada automaticamente pelo FastAPI, onde você pode testar o endpoint `/api/v1/inference`.

---

## Resultados

Relatório de classificação gerado após o treinamento do modelo:

```markdown
Relatório de classificação:
              precision    recall  f1-score   support

           0       0.92      0.87      0.90      7592
           1       0.86      0.88      0.87     10051
           2       0.90      0.92      0.91      7748

    accuracy                           0.89     25391
   macro avg       0.90      0.89      0.89     25391
weighted avg       0.89      0.89      0.89     25391
```
