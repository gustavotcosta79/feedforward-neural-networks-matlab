[inputsTest, targetsTest] = convertImage_draw();

%carregar as melhores redes
load('rede_top_98_005117.mat', 'netAtual');
net1 = netAtual;
load('rede_top_95_005203.mat', 'netAtual');
net2 = netAtual;
load('rede_top_70_005231.mat', 'netAtual');
net3 = netAtual;

% === Avaliar a Rede 1 ===
outputs1 = net1(inputsTest);
[~, pred1] = max(outputs1);
[~, trueLabels] = max(targetsTest);
acc1 = sum(pred1 == trueLabels) / length(trueLabels);
fprintf('Rede 1 - Precisão TESTE: %.2f%%\n', acc1 * 100);
figure; plotconfusion(targetsTest, outputs1);
title('Matriz de Confusão - Rede 1');


% === Avaliar a Rede 2 ===
outputs2 = net2(inputsTest);
[~, pred2] = max(outputs2);
acc2 = sum(pred2 == trueLabels) / length(trueLabels);
fprintf('Rede 2 - Precisão TESTE: %.2f%%\n', acc2 * 100);
figure; plotconfusion(targetsTest, outputs2);
title('Matriz de Confusão - Rede 2');


% === Avaliar a Rede 3 ===
outputs3 = net3(inputsTest);
[~, pred3] = max(outputs3);
acc3 = sum(pred3 == trueLabels) / length(trueLabels);
fprintf('Rede 3 - Precisão TESTE: %.2f%%\n', acc3 * 100);
figure; plotconfusion(targetsTest, outputs3);
title('Matriz de Confusão - Rede 3');









