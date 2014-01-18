close all;
clear all;
Base = [0.8834;

0.8584;

0.8520;

0.8499];



m1 = [0.8797;

0.8597;

0.8526;

0.8501];



m2 = [

0.8793;

0.8583;

0.8533;

0.8499];

r = [3,5,7,10];



figure;

hold on;
axis auto;
plot(r,Base,'r.-','Linewidth',2);

plot(r,m1,'b.-','Linewidth',2);

plot(r,m2,'g.-','Linewidth',2);

xl = xlabel('Rank');

yl = ylabel('Prediction RMSE averaged over 20 trials');

LEG  = legend('Baseline','Method-1','Method-2');
set(xl, 'FontSize', 14);
set(yl, 'FontSize', 14);
set(LEG, 'FontSize', 18);
ylim([0.847,0.885]);
set(gca,'fontsize',14)
print -f1 -djpeg -r600 MovieLens1MGenre.jpg