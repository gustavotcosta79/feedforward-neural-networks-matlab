clear; clc;
% === Carregar os dados ===
[inputsStart, targetsStart] = convertImage_start();
[inputsTrain, targetsTrain] = convertImage_train();
[inputsTest, targetsTest] = convertImage_test();

% === Juntar todos os dados ===
inputsAll = [inputsStart, inputsTrain, inputsTest];
targetsAll = [targetsStart, targetsTrain, targetsTest];

%carregar as melhores redes
load('rede_top_76_run00_211240.mat ', 'net');
net1 = net;
load('rede_top_78_run00_211342.mat', 'net');
net2 = net;
load('rede_top_78_run00_211643.mat ', 'net');
net3 = net;


% Lista com as redes e nomes
redes = {net1, net2, net3};
nomes = {'Rede 1', 'Rede 2', 'Rede 3'};

% === Para cada rede ===
for i = 1:length(redes)
    fprintf('\n--- %s ---\n', nomes{i});
    netAtual = redes{i};

    % Configurar divisão dos dados (mesmo que seja todos juntos)
    netAtual.divideParam.trainRatio = 0.8;
    netAtual.divideParam.valRatio = 0.1;
    netAtual.divideParam.testRatio = 0.1;

    % === Treinar com todos os dados ===
    [netAtual, tr] = train(netAtual, inputsAll, targetsAll);

      % === PRECISAO GLOBAL ===
    outputsAll = netAtual(inputsAll);
    [~, predAll] = max(outputsAll);       % Classificações da rede
    [~, trueAll] = max(targetsAll);       % Labels reais

    accGlobal = sum(predAll == trueAll) / length(trueAll);
    fprintf('Precisão GLOBAL: %.2f%%\n', accGlobal * 100);

    % === PRECISAO DE TESTE ===
    idxTest = tr.testInd;
    predTest = predAll(idxTest);
    trueTest = trueAll(idxTest);

    accTeste = sum(predTest == trueTest) / length(idxTest);
    fprintf('Precisão TESTE: %.2f%%\n', accTeste * 100);

    % === CLASSIFICAR START ===
    outputsStart = netAtual(inputsStart);
    [~, predStart] = max(outputsStart);
    [~, trueStart] = max(targetsStart);
    accStart = sum(predStart == trueStart) / length(trueStart);
    fprintf('Precisão START: %.2f%%\n', accStart * 100);
    
    % === Matriz de Confusão (TRAIN) ===
    figure;
    plotconfusion(targetsStart, outputsStart);
    title(['Matriz de Confusão - ', nomes{i}, ' (START)']);

    % === CLASSIFICAR TRAIN ===
    outputsTrain = netAtual(inputsTrain);
    [~, predTrain] = max(outputsTrain);
    [~, trueTrain] = max(targetsTrain);
    accTrain = sum(predTrain == trueTrain) / length(trueTrain);
    fprintf('Precisão TRAIN: %.2f%%\n', accTrain * 100);

     % === Matriz de Confusão (TRAIN) ===
    figure;
    plotconfusion(targetsTrain, outputsTrain);
    title(['Matriz de Confusão - ', nomes{i}, ' (TRAIN)']);

    % === CLASSIFICAR TEST ===
    outputsTest = netAtual(inputsTest);
    [~, predTest] = max(outputsTest);
    [~, trueTest] = max(targetsTest);
    accTest = sum(predTest == trueTest) / length(trueTest);
    fprintf('Precisão TEST: %.2f%%\n', accTest * 100);

    % === Matriz de Confusão (TEST) ===
    figure;
    plotconfusion(targetsTest, outputsTest);
    title(['Matriz de Confusão - ', nomes{i}, ' (TEST)']);


    % Gravar rede com precisão de teste
    nomeFicheiro = sprintf('rede_top_%02d_%s.mat', round(accTest * 100), datestr(now,'HHMMSS'));
    save(nomeFicheiro, 'netAtual');  % Salvar a rede com o nome gerado
    fprintf('Rede guardada como "%s" com %.2f%% de precisão de teste.\n', nomeFicheiro, accTest * 100);
end