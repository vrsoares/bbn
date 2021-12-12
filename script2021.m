
%%%%%%%%%%%% PAR�METROS INICIAIS %%%%%%%%%%%%%%%%%%
N = 8; 
dag = zeros(N,N);
%C = 1; IBL = 2; AP = 3;
vaccination_policy = 1;
people_fully_vaccinated_per_hundred = 2;
income_support = 3;
contact_tracing = 4;
testing_policy = 5;
stay_home_requirements = 6;
facial_coverings = 7;
deaths_millions_diff = 8;


discrete_nodes = 1:N;
node_sizes = [6,8,3,3,4,4,5,20];



%%%%%%%%%%%% REDE ORIGINAL COM TCP's DADAS PARA CADA Nó (dá para usar para comparar com os métodos de estimação %%%%%%%%%%%%%%%%%%
% bnet = mk_bnet(dag, node_sizes, 'names', {'C','IBL','AP'}, 'discrete', 1:3);
% 
% bnet.CPD{C} = tabular_CPD(bnet, C, [0.25 0.75]);
% bnet.CPD{IBL} = tabular_CPD(bnet, IBL, [0.75 0.25]);  
% bnet.CPD{AP} = tabular_CPD(bnet, AP, [1 0.6 0 0 0 0.4 1 1]);
CPT = cell(1,N);

CPT{1,vaccination_policy} = [   0
                                0.04
                                0.12
                                0.22
                                0.3
                                0.32];

CPT{1,people_fully_vaccinated_per_hundred} = [  0	0	0	0	0	0	0	0
                                                0.02	0.02	0	0	0	0	0	0
                                                0.08	0.02	0.02	0	0	0	0	0
                                                0.02	0.08	0.08	0.04	0	0	0	0
                                                0	0	0.02	0.06	0.1	0.04	0.02	0.06
                                                0	0	0	0.02	0.04	0.1	0.14	0.02];

CPT{1,income_support} = [   0.02	0.16	0.16
                            0.02	0.04	0.18
                            0	0	0.08
                            0.04	0	0.2
                            0.02	0.06	0.02];

CPT{1,contact_tracing} = [  0.12	0.14	0.08
                            0.02	0.14	0.08
                            0	0.04	0.04
                            0	0.02	0.22
                            0.04	0.04	0.02];

CPT{1,testing_policy} = [   0	0.12	0	0.22
                            0	0.02	0.08	0.14
                            0	0	0.06	0.02
                            0	0.02	0.1	0.12
                            0	0.04	0	0.06];

CPT{1,stay_home_requirements} = [   0	0.06	0.24	0.04
                                    0	0.12	0.08	0.04
                                    0	0.06	0.02	0
                                    0.02	0.14	0.02	0.06
                                    0	0.04	0	0.06];

CPT{1,facial_coverings} = [ 0	0	0.06	0.12	0.16
                            0	0.02	0.1	0.08	0.04
                            0	0.02	0.06	0	0
                            0	0.06	0.12	0.04	0.02
                            0	0	0.02	0.04	0.04];

CPT{1,deaths_millions_diff} = [ 0.02	0	0	0	0	0	0	0	0.02	0	0	0	0	0.02	0	0.02	0.02	0	0	0.02
                                0.04	0	0	0	0	0	0	0	0.04	0	0	0	0	0	0.02	0	0	0	0.02	0
                                0.04	0	0	0	0	0.02	0	0.02	0.02	0	0	0.02	0	0	0	0	0	0	0	0
                                0.02	0.02	0.02	0.02	0	0	0	0	0	0	0.02	0	0	0	0	0	0	0.02	0	0
                                0	0	0	0.02	0.04	0.02	0.02	0.02	0	0	0	0	0	0	0.02	0	0	0	0	0
                                0	0	0	0	0.02	0.02	0	0	0.02	0.02	0	0.02	0	0	0.02	0	0	0.02	0	0
                                0	0.04	0	0.02	0	0	0	0	0	0	0	0	0	0	0.02	0.02	0.02	0	0.02	0.02
                                0	0	0.02	0	0	0	0	0.02	0	0	0	0	0	0	0.02	0	0.02	0	0	0];

% Dados Originais
dados=readtable('2021.csv');
dados = table2array(dados);
dados=dados';
 %2 � um e 1 � zero
 
%  CPT = cell(1,N);
% for i=1:N
%   s=struct(bnet.CPD{i});  % violate object privacy
%   CPT{i}=s.CPT;
% end
% 
% % celldisp(CPT)
% dispcpt(CPT{1})
% dispcpt(CPT{2})
% dispcpt(CPT{3})

%%%%%%%%%%%%%%%%%%%%%%% ESTIMANDO A ESTRUTURA DA REDE (SUPOR QUE NÃO SABEMOS A
%%%%%%%%%%%%%%%%%%%%%%% LIGAÇÃO DAS ARESTAS)

%Montar a estrutura da rede atrav�s da m�trica K2 - aqui o algoritmo tenta
%encontrar a melhor estrutura de rede
% order = [
%          new_deaths_previous
%          facial_coverings
%          stay_home_requirements
%          income_support
%          contact_tracing
%          testing_policy
%          vaccination_policy
%          new_deaths
% ];
% clamped = zeros(N, size(dados,2));
% clamped(1:N-1,:) = 1; %esse deixa so a saida
root = 1;
dag_best  = learn_struct_mwst(dados,ones(1,N), node_sizes, 'tabular', 'mutual_info',root); %root=4
% for i=2:N-1
%     dag(1,i) = 1;
% end
% dag(2:N-1,N) = 1;
% dag_best = dag;
bnet1 = mk_bnet(dag_best, node_sizes);
G1 = bnet1.dag
draw_graph(G1)
title('Estrutura encontrada pela métrica para os dados originais')



% Cria��o das TPC's (Tabela propbabilidade condicional) randomicas atrav�s da priori conjugada da multinomial, a
% Dirichlet: theta_ijk = (N_ijk + alpha_ijk) / (N_ij + alpha_ij): n�o
% importa ser random os parametros iniciais

bnet1.CPD{vaccination_policy}=tabular_CPD(bnet1, vaccination_policy, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{people_fully_vaccinated_per_hundred}=tabular_CPD(bnet1, people_fully_vaccinated_per_hundred, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{income_support}=tabular_CPD(bnet1, income_support, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{contact_tracing}=tabular_CPD(bnet1, contact_tracing, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{testing_policy}=tabular_CPD(bnet1, testing_policy, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{stay_home_requirements}=tabular_CPD(bnet1, stay_home_requirements, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{facial_coverings}=tabular_CPD(bnet1, facial_coverings, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');
bnet1.CPD{deaths_millions_diff}=tabular_CPD(bnet1, deaths_millions_diff, 'prior_type', 'dirichlet', 'dirichlet_type', 'BDeu');


ncases = 1000;
data = zeros(N, ncases);
for m=1:ncases
  data(:,m) = cell2num(sample_bnet(bnet1));% Amostragem direta da rede bnet1 
end
samples=data;
% 
% 
% dag_best2 = learn_struct_K2(samples, node_sizes, order, 'verbose', 'yes')%n�o retorna as TPC's, somente a estrutura
% bnet5 = mk_bnet(dag_best2, node_sizes);
% G2 = bnet5.dag
% figure
% draw_graph(G2)

%%%%%%%% comparando estimação doa parametros por maxima verossimilhanca e por
% inferencia bayseiana %%%%%%%%%%%%%%%%

 bnet2 = learn_params(bnet1, samples);% estima�ao por maxima verossimilhan�a para dados completos
 
 % Percebe-se que os valores encontrados pelos estimadores das  CTP1 e CTP2
 % s�o muito pr�ximos aos valores reais CTP

 clamped = zeros(N, ncases);
 bnet3=bayes_update_params(bnet1, samples, clamped); %estima�ao por inferencia bayesiana para dados completos


%%%%%%%%%%%%%%%%%%%%%%% VISUALIZA��O DAS CTP's%%%%%%%%%%%%%%%%%%%%%
%To view the learned parameters, we use a little Matlab hackery.
CPT1 = cell(1,N);
for i=1:N
  s=struct(bnet1.CPD{i});  % violate object privacy
  CPT1{i}=s.CPT;
end

CPT2 = cell(1,N);
for i=1:N
  s=struct(bnet2.CPD{i});  % violate object privacy
  CPT2{i}=s.CPT;
end

CPT3 = cell(1,N);
for i=1:N
  s=struct(bnet3.CPD{i});  % violate object privacy
  CPT3{i}=s.CPT;
end

fprintf('\nPEOPLE FULLY VACCINATED SE VACCINATION POLICY NIVEL 6\n')
fprintf('\nOriginal\n')
disp(CPT{1,people_fully_vaccinated_per_hundred}(6,:))
fprintf('\nCPT1\n')
disp(CPT1{1,people_fully_vaccinated_per_hundred}(6,:))
fprintf('\nCPT3\n')
disp(CPT3{1,people_fully_vaccinated_per_hundred}(6,:))

fprintf('\n DEATH DIFF SE PEOPLE FULLY VACCINATED NIVEL 6, 7 e 8\n')
fprintf('\nOriginal\n')
soma = sum(CPT{1,deaths_millions_diff}(6:8,:),1);
disp(soma)
fprintf('\nCPT1\n')
soma = sum(CPT1{1,deaths_millions_diff}(6:8,:),1);
disp(soma/3)
fprintf('\nCPT3\n')
soma = sum(CPT3{1,deaths_millions_diff}(6:8,:),1);
disp(soma/3)

fprintf('\n STAY HOME se DEATH DIFF 1 a 9\n')
fprintf('\nOriginal\n')
soma = sum(CPT{1,stay_home_requirements}(1:2,:),1);
disp(soma)
fprintf('\nCPT1\n')
soma = sum(CPT1{1,stay_home_requirements}(1:9,:),1);
disp(soma/9)
fprintf('\nCPT3\n')
soma = sum(CPT3{1,stay_home_requirements}(1:9,:),1);
disp(soma/9)

fprintf('\n MASK USE se DEATH DIFF 1 a 9\n')
fprintf('\nOriginal\n')
soma = sum(CPT{1,facial_coverings}(1:2,:),1);
disp(soma)
fprintf('\nCPT1\n')
soma = sum(CPT1{1,facial_coverings}(1:9,:),1);
disp(soma/9)
fprintf('\nCPT3\n')
soma = sum(CPT3{1,facial_coverings}(1:9,:),1);
disp(soma/9)


% celldisp(CPT3)
 
%Here are the parameters learned for nodes
% dispcpt(CPT1{1})
% dispcpt(CPT1{2})
% dispcpt(CPT1{3})
% dispcpt(CPT1{4})
% dispcpt(CPT1{5})
% dispcpt(CPT1{6})
% dispcpt(CPT1{7})
% dispcpt(CPT1{8})
% 
% dispcpt(CPT2{1})
% dispcpt(CPT2{2})
% dispcpt(CPT2{3})
% dispcpt(CPT2{4})
% dispcpt(CPT2{5})
% dispcpt(CPT2{6})
% dispcpt(CPT2{7})
% dispcpt(CPT2{8})
% 
% %Mostrar os param estimados por inferencia bayesiana
% dispcpt(CPT3{1})
% dispcpt(CPT3{2})
% dispcpt(CPT3{3})
% dispcpt(CPT3{4})
% dispcpt(CPT3{5})
% dispcpt(CPT3{6})
% dispcpt(CPT3{7})
% dispcpt(CPT3{8})

% Para conferir se o learn_struct calcula certinnho os thetas igual CPT
% original:

% bnet4 = learn_params(bnet, dados)
% 
% CPT4 = cell(1,N);
% for i=1:N
%   s=struct(bnet4.CPD{i});  % violate object privacy
%   CPT4{i}=s.CPT;
% end
% 
% dispcpt(CPT4{1})
% dispcpt(CPT4{2})
% dispcpt(CPT4{3})



%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AMOSTRAGEM PARA GERA��O DA NOVA POPULA��O %%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%

% %Gera��o do conjunto de dados. Amostras da rede bnet1
% ncases = 100;
% data = zeros(N, ncases);
% for m=1:ncases
%   data(:,m) = cell2num(sample_bnet(bnet2));% Amostragem direta da rede bnet1 
% end



%Gera��o do conjunto de dados. Amostras da rede bnet para estima��o dos
%par�metros - Popula��o Inicial que pode ser aleat�ria
% ncases = 100;
% data = zeros(N, ncases);
% for m=1:ncases
%   data(:,m) = cell2num(sample_bnet(bnet1));% Amostragem direta da rede bnet1 
% end
% samples=data;



%Referencia: http://bnt.googlecode.com/svn/trunk/docs/usage.html#basics
