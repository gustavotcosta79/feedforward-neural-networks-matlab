clear; clc;

% === Carregar imagens da pasta train ===
[inputs, targets] = convertImage_train();

% === Configuração da Rede ===
topologia = [10 10];                   
funcTreino = 'trainlm';                
funcAtivacao = 'tansig';              
ratios = [70 15 15];                   

fprintf('\nTopologia: [%s]\n', num2str(topologia));
fprintf('Função de treino: %s\n', funcTreino);
fprintf('Função de ativação: %s\n', funcAtivacao);
fprintf('Ratios: treino %d%% | validação %d%% | teste %d%%\n', ratios);

% === Criar rede ===
net = feedforwardnet(topologia, funcTreino);

% Definir funções de ativação para todas as camadas escondidas
for i = 1:length(net.layers) - 1
    net.layers{i}.transferFcn = funcAtivacao;
end

% Número de épocas 
net.trainParam.epochs = 20;  

% Rácio de divisão treino/validação/teste
net.divideParam.trainRatio = ratios(1)/100;
net.divideParam.valRatio   = ratios(2)/100;
net.divideParam.testRatio  = ratios(3)/100;

% === Treinar ===
[net, tr] = train(net, inputs, targets);

% === Avaliação GLOBAL ===
outputs = net(inputs);
[~, predClasses] = max(outputs);
[~, trueClasses] = max(targets);
accuracyGlobal = sum(predClasses == trueClasses) / length(trueClasses);

% === Avaliação TESTE ===
testInd = tr.testInd;
testOutputs = net(inputs(:, testInd));
[~, testPred] = max(testOutputs);
[~, testTrue] = max(targets(:, testInd));
accuracyTeste = sum(testPred == testTrue) / length(testTrue);

fprintf('Precisão GLOBAL: %.2f%%\n', accuracyGlobal * 100);
fprintf('Precisão TESTE:  %.2f%%\n', accuracyTeste * 100);

% === Matriz de confusão
figure;
plotconfusion(ind2vec(testTrue), ind2vec(testPred));
title(sprintf('Confusion Matrix (TESTE) - Topologia [%s]', num2str(topologia)));

% === Gravar rede ===
nomeFicheiro = sprintf('rede_top_%02d_%s.mat', round(accuracyTeste*100), datestr(now,'HHMMSS'));
save(nomeFicheiro, 'net');
fprintf('Rede guardada como "%s" com %.2f%% de precisão de teste.\n', nomeFicheiro, accuracyTeste * 100);
