% Obter índices de treino/validação/teste automaticamente
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

[net, tr] = train(net, inputs, targets);

% Avaliar desempenho global (com todos os dados)
outputs = net(inputs);
[~, predClasses] = max(outputs);
[~, trueClasses] = max(targets);
accuracy = sum(predClasses == trueClasses) / length(trueClasses);
fprintf('Precisão global: %.2f%%\n', accuracy * 100);

% Avaliar desempenho no conjunto de teste
testInd = tr.testInd;
testOutputs = net(inputs(:, testInd));
[~, testPred] = max(testOutputs);
[~, testTrue] = max(targets(:, testInd));

% Mostrar matriz de confusão 
figure;
plotconfusion(ind2vec(testTrue), ind2vec(testPred));
title(sprintf('Confusion Matrix (TESTE) - Topologia [%s]', num2str(topologia)));
