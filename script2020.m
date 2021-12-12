
%%%%%%%%%%%% PARAMETROS INICIAIS %%%%%%%%%%%%%%%%%%
%Numeros de variaveis
N = 6; 

%Associando números a variáveis
income_support = 1;
contact_tracing = 2;
testing_policy = 3;
stay_home_requirements = 4;
facial_coverings = 5;
new_deaths = 6;


%Criando nós a partir das variáveis
discrete_nodes = 1:N;

%Listando valores que podem assumir cada uma das variavies, em ordem
%crescente a partir da associacao feita 2 etapas atras
node_sizes = [3,3,4,4,5,9];

%%%%%%%%%%%% INSERCAO DE DADOS E CPT'S ORIGINAIS %%%%%%%%%%%%%%%%%%
% Dados Originais
dados=readtable('2020.csv');
dados = table2array(dados);
dados=dados';

%CPT Original
CPT = cell(1,N);
CPT{1,contact_tracing} = [  0.1
                            0.45
                            0.45];

CPT{1,income_support} = [   0.1	        0	        0
                            0.083333333	0.016666667	0.35
                            0.016666667	0.183333333	0.25];


CPT{1,stay_home_requirements} = [   0.166666667	0.033333333	0	0
                                    0	0.016666667	0.15	0.033333333
                                    0.166666667	0.133333333	0.183333333	0.116666667];

CPT{1,new_deaths} = [   0.3	0.016666667	0.016666667	0	0	0	0 0 0
                        0.033333333	0	0.033333333	0.066666667	0.016666667	0.033333333	0 0 0
                        0	0	0.033333333	0.016666667	0.016666667	0.216666667	0.05 0 0
                        0	0	0	0	0.016666667	0.1	0.033333333 0 0];


CPT{1,testing_policy} = [   0.067	0.133	0.000	0.000
                            0.000	0.017	0.167	0.017
                            0.000	0.100	0.217	0.283];

CPT{1,facial_coverings} = [ 0.183333333	0	0.016666667	0	0
                            0	0.016666667	0.033333333	0	0.15
                            0.1	0.083333333	0.25	0.066666667	0.1];
%%%%%%%%%%%%%%%%%%%%%%% ESTIMANDO A ESTRUTURA DA REDE (SUPOR QUE NÃO SABEMOS A
%%%%%%%%%%%%%%%%%%%%%%% LIGAÇÃO DAS ARESTAS)

%Greedy
dag_best  = learn_struct_gs2(dados, node_sizes);
dag_best(4,6) = 1;
bnet1 = mk_bnet(dag_best, node_sizes);
G1 = bnet1.dag
draw_graph(G1)
title('Estrutura encontrada pela métrica para os dados originais')



% Cria��o das TPC's (Tabela propbabilidade condicional) randomicas atrav�s da priori conjugada da multinomial, a
% Dirichlet: theta_ijk = (N_ijk + alpha_ijk) / (N_ij + alpha_ij): n�o
% importa ser random os parametros iniciais

bnet1.CPD{income_support}=tabular_CPD(bnet1, income_support, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{contact_tracing}=tabular_CPD(bnet1, contact_tracing, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{testing_policy}=tabular_CPD(bnet1, testing_policy, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{stay_home_requirements}=tabular_CPD(bnet1, stay_home_requirements, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{facial_coverings}=tabular_CPD(bnet1, facial_coverings, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{new_deaths}=tabular_CPD(bnet1, new_deaths, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');


ncases = 1000;
data = zeros(N, ncases);
for m=1:ncases
  data(:,m) = cell2num(sample_bnet(bnet1));% Amostragem direta da rede bnet1 
end
samples=data;
% 
% 
%%%%%%%% comparando estimação doa parametros por maxima verossimilhanca e por
% inferencia bayseiana %%%%%%%%%%%%%%%%

 %bnet2 = learn_params(bnet1, samples);% estima�ao por maxima verossimilhan�a para dados completos
 
 % Percebe-se que os valores encontrados pelos estimadores das  CTP1 e CTP2
 % s�o muito pr�ximos aos valores reais CTP

 clamped = zeros(N, ncases);
%  clamped(1:N-1,:) = 1;
 bnet3=bayes_update_params(bnet1, samples, clamped); %estima�ao por inferencia bayesiana para dados completos


%%%%%%%%%%%%%%%%%%%%%%% VISUALIZA��O DAS CTP's%%%%%%%%%%%%%%%%%%%%%
%To view the learned parameters, we use a little Matlab hackery.
CPT1 = cell(1,N);
for i=1:N
  s=struct(bnet1.CPD{i});  % violate object privacy
  CPT1{i}=s.CPT;
end

% CPT2 = cell(1,N);
% for i=1:N
%   s=struct(bnet2.CPD{i});  % violate object privacy
%   CPT2{i}=s.CPT;
% end

CPT3 = cell(1,N);
for i=1:N
  s=struct(bnet3.CPD{i});  % violate object privacy
  CPT3{i}=s.CPT;
end

fprintf('\nMORTE SE STAY_AT_HOME NIVEL 3\n')
fprintf('\nOriginal\n')
disp(CPT{1,new_deaths}(3,:))
fprintf('\nCPT1\n')
disp(CPT1{1,new_deaths}(3,:))
fprintf('\nCPT3\n')
disp(CPT3{1,new_deaths}(3,:))


fprintf('\nSTAY_AT_HOME SE INCOME NIVEL 3\n')
fprintf('\nOriginal\n')
disp(CPT{1,stay_home_requirements}(3,:))
fprintf('\nCPT1\n')
disp(CPT1{1,stay_home_requirements}(3,:))
fprintf('\nCPT3\n')
disp(CPT3{1,stay_home_requirements}(3,:))
% celldisp(CPT3)
 
% %Here are the parameters learned for nodes
% fprintf('\nCPT1\n')
% dispcpt(CPT1{1})
% dispcpt(CPT1{2})
% dispcpt(CPT1{3})
% dispcpt(CPT1{4})
% dispcpt(CPT1{5})
% dispcpt(CPT1{6})
% 
% fprintf('\nCPT2\n')
% dispcpt(CPT2{1})
% dispcpt(CPT2{2})
% dispcpt(CPT2{3})
% dispcpt(CPT2{4})
% dispcpt(CPT2{5})
% dispcpt(CPT2{6})
% 
% %Mostrar os param estimados por inferencia bayesiana
% fprintf('\nCPT3\n')
% dispcpt(CPT3{1})
% dispcpt(CPT3{2})
% dispcpt(CPT3{3})
% dispcpt(CPT3{4})
% dispcpt(CPT3{5})
% dispcpt(CPT3{6})



% % Para conferir se o learn_struct calcula certinnho os thetas igual CPT
% % original:
% 
% bnet4 = learn_params(bnet, dados)
% 
% CPT4 = cell(1,N);
% for i=1:N
%   s=struct(bnet4.CPD{i});  % violate object privacy
%   CPT4{i}=s.CPT;
% end
% % 
% % dispcpt(CPT4{1})
% % dispcpt(CPT4{2})
% % dispcpt(CPT4{3})
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% AMOSTRAGEM PARA GERA��O DA NOVA POPULA��O %%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%
% 
% % %Gera��o do conjunto de dados. Amostras da rede bnet1
% % ncases = 100;
% % data = zeros(N, ncases);
% % for m=1:ncases
% %   data(:,m) = cell2num(sample_bnet(bnet2));% Amostragem direta da rede bnet1 
% % end
% 
% 
% 
% %Gera��o do conjunto de dados. Amostras da rede bnet para estima��o dos
% %par�metros - Popula��o Inicial que pode ser aleat�ria
% % ncases = 100;
% % data = zeros(N, ncases);
% % for m=1:ncases
% %   data(:,m) = cell2num(sample_bnet(bnet1));% Amostragem direta da rede bnet1 
% % end
% % samples=data;
% 
% 
% 
% %Referencia: http://bnt.googlecode.com/svn/trunk/docs/usage.html#basics
