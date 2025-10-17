# Projeto Iris - DL4J

## Visão geral
Este projeto treina uma rede neural com Deeplearning4j sobre o dataset Iris. O pipeline faz leitura de um arquivo(.data), transforma o rótulo categórico em inteiro, normaliza os dados, divide em treino/teste e treina um `MultiLayerNetwork` com parada precoce quando a acurácia de teste atingir `1.0`.

## Estrutura de arquivos
- `src/main/java/org/thiago/Main.java` — código principal (pipeline, modelo e loop de treino).
- `src/main/resources/iris.data` — arquivo (CSV ou .data) do dataset Iris (4 features + rótulo textual).
- `build.gradle` — dependências e configuração Gradle.

## Requisitos
- Java 17
- Gradle (ou usar `gradlew.bat` no Windows)
- Dependências: Deeplearning4j, ND4J, DataVec (definidas em `build.gradle`)

## O que o código faz (passos principais)
1. Semente determinística:
   - `Nd4j.getRandom().setSeed(seed)` garante reprodutibilidade parcial.
2. Leitura e schema:
   - Define schema com 4 colunas double (`sepallength`, `sepalwidth`, `petallength`, `petalwidth`) e 1 coluna string `label`.
3. Transformações:
   - `StringToCategoricalTransform("label", categories)` converte texto em categoria.
   - `CategoricalToIntegerTransform("label")` converte categoria em inteiro (0..2).
   - Uso de `TransformProcessRecordReader` para aplicar as transformações ao ler o CSV.
4. Iterator:
   - `RecordReaderDataSetIterator` monta um único `DataSet` com `batchSize = 150` e `labelIndex = 4`.
5. Normalização:
   - `NormalizerStandardize` (fit no `allData` e transform) para média 0 / variância 1.
6. Split treino/teste:
   - `allData.splitTestAndTrain(0.8)` → 80% treino / 20% teste.
7. Arquitetura do modelo:
   - Rede MLP com duas camadas Dense e uma camada de saída Softmax:
     - Entrada: 4
     - Hidden: 50, 50 (ReLU)
     - Output: 3 (Softmax)
   - Otimizador: Adam com `lr = 0.005`
   - Inicialização: `WeightInit.XAVIER`
8. Treinamento:
   - Loop por até `maxEpochs` (10000).
   - A cada época os dados de treino são embaralhados (`trainingData.shuffle(seed + epoch)`).
   - A cada 10 épocas é calculada a acurácia no conjunto de teste; se `acc == 1.0` ocorre parada precoce.
9. Avaliação final:
   - Usa `Evaluation` para imprimir métricas e matriz de confusão.

## Interpretação dos logs / métricas
- `ScoreIterationListener` imprime loss (score) a cada 100 iterações.
- Métricas impressas por `Evaluation`:
  - Accuracy, Precision, Recall, F1 (macro-averaged).
  - Matriz de confusão: linhas = classes reais, colunas = classes previstas.
- Exemplo obtido: acurácia de teste chegou a `1.0` na época 20 no ambiente de execução.

## Overfitting — como detectar aqui
- Ter acurácia 1.0 no teste não prova overfitting; pode indicar que o dataset é facilmente separável (Iris) e o split também foi favorável.
- Verificar:
  - Comparar métricas de treino vs teste por época (train acc / train loss vs test acc / test loss).
  - Se `train acc >> test acc` ou `train loss` muito baixo enquanto `test loss` alto → overfitting.
- Código sugerido (já indicado no projeto): registrar `model.score(trainingData)` e `accuracy` do treino a cada época para comparação.

## Como reduzir/evitar overfitting
- Regularização L2: adicionar `.l2(1e-4)` nas camadas.
- Dropout: `.dropOut(0.5)` nas camadas Dense.
- Reduzir capacidade: menor número de neurônios / camadas.
- EarlyStopping baseado em loss/validação (usar `EarlyStoppingTrainer` de DL4J).
- Cross-validation / k-fold ou usar um conjunto de validação separado.

## Hiperparâmetros chave (valores atuais)
- seed = `12345`
- batchSize = `150`
- learning rate (Adam) = `0.005`
- numHidden = `50`
- maxEpochs = `10000`
- split treino = `0.8`

## Notas e avisos
- Aviso de AVX: ND4J pode imprimir warning sobre AVX/AVX2; isso não impede execução, só indica potencial de otimização.
- O comportamento reprodutível depende de semente e das configurações internas do backend; resultados idênticos entre máquinas nem sempre são garantidos.
- O dataset Iris é pequeno e simples — atingir 1.0 de acurácia é plausível sem overfitting severo.

## Sugestões rápidas para experimentação
- Testar L2 + dropout se quiser reduzir chance de overfitting.
- Monitorar curvas (loss/accuracy) por época e salvar checkpoints do modelo.

## Resultado esperado
- Logs mostrando loss e acurácia.
