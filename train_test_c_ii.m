clear; clc;

% === Carregar conjuntos de dados ===
[inputsStart, targetsStart] = convertImage_start();
[inputsTrain, targetsTrain] = convertImage_train();
[inputsTest, targetsTest] = convertImage_test();

% Carregar as melhores redes
load('rede_top_76_run00_211240.mat', 'net');
net1 = net;
load('rede_top_78_run00_211342.mat', 'net');
net2 = net;
load('rede_top_78_run00_211643.mat', 'net');
net3 = net;

% Lista com as redes
redes = {net1, net2, net3};
nomes = {'Rede 1', 'Rede 2', 'Rede 3'};

% === Para cada rede ===
for i = 1:length(redes)
    fprintf('\n--- %s ---\n', nomes{i});
    netAtual = redes{i};

    % Divisão dos dados para treino
    netAtual.divideParam.trainRatio = 0.7;
    netAtual.divideParam.valRatio   = 0.15;
    netAtual.divideParam.testRatio  = 0.15;

    % Treinar com dados da pasta TEST
    [netAtual, tr] = train(netAtual, inputsTest, targetsTest);

    % === Avaliação com START ===
    outputsStart = netAtual(inputsStart);
    [~, predStart] = max(outputsStart);
    [~, trueStart] = max(targetsStart);
    accStart = sum(predStart == trueStart) / length(trueStart);
    fprintf('>>> Precisão de Teste usando imagens da PASTA START: %.2f%%\n', accStart * 100);
    
    figure;
    numClasses = size(targetsStart, 1);
    plotconfusion(full(ind2vec(trueStart, numClasses)), full(ind2vec(predStart, numClasses)));
    title(['Matriz de Confusão - ', nomes{i}, ' (START)']);

    % === Avaliação com TRAIN ===
    outputsTrain = netAtual(inputsTrain);
    [~, predTrain] = max(outputsTrain);
    [~, trueTrain] = max(targetsTrain);
    accTrain = sum(predTrain == trueTrain) / length(trueTrain);
    fprintf('>>> Precisão de Teste usando imagens da PASTA TRAIN: %.2f%%\n', accTrain * 100);
    
    figure;
    numClasses = size(targetsTrain, 1);
    plotconfusion(full(ind2vec(trueTrain, numClasses)), full(ind2vec(predTrain, numClasses)));
    title(['Matriz de Confusão - ', nomes{i}, ' (TRAIN)']);

    % === Avaliação com TEST ===
    outputsTest = netAtual(inputsTest);
    [~, predTest] = max(outputsTest);
    [~, trueTest] = max(targetsTest);
    accTest = sum(predTest == trueTest) / length(trueTest);
    fprintf('>>> Precisão de Teste usando imagens da PASTA TEST: %.2f%%\n', accTest * 100);
    
    figure;
    numClasses = size(targetsTest, 1);
    plotconfusion(full(ind2vec(trueTest, numClasses)), full(ind2vec(predTest, numClasses)));
    title(['Matriz de Confusão - ', nomes{i}, ' (TEST)']);

end
