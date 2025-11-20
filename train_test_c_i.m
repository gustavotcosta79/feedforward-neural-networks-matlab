[inputsTest, targetsTest] = convertImage_test();

%carregar as melhores redes
load('rede_top_76_run00_211240.mat ', 'net');
net1 = net;
load('rede_top_78_run00_211342.mat', 'net');
net2 = net;
load('rede_top_78_run00_211643.mat ', 'net');
net3 = net;


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









