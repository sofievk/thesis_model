%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%      Section 1: Parameters        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Time%%
%%%%%%%%
T = 30;             %Number of direct optimization periods T
y = (1:1:T);        %Corresponding calendar years    
y(1) = 2010;
for i = 1:1:T-1;
    y(1+i) = 2010+((i)*10);
end
n = 100;            %Number of pre-balanced growth path simulation periods after T
y2 = (1:1:T+n);     %Corresponding calendar years   
y2(1) = 2010;
   for i = 1:1:T-1+n;
       y2(1+i) = 2010+((i)*10);
   end

%%Climate and Damages%%
%%%%%%%%%%%%%%%%%%%%%%%
phi = 0.0228;       %Carbon depreciation per annum (remaining share)
phiL = 0.2;         %Carbon emitted to the atmosphere staying there forever
phi0 = 0.393;       %Share of remaining emissions exiting atmosphere immediately
Sbar = 581;         %Pre-industrial atmospheric GtC
S1_2000 = 103;      %GtC
S2_2000 = 699;      %GtC
gamma = zeros(T,1); 
for i = 1:1:T;
    gamma(i) = 0.000023793; %Damage elasticity
end
 
%%Energy Aggregation%%
%%%%%%%%%%%%%%%%%%%%%%
rho = -0.058;      %Elasticity of substitution between energy sources
kappa1 = 0.5429;   %Relative efficiency of oil
kappa2 = 0.1015;   %Relative efficiency of coal
kappa3 = 1-kappa1-kappa2; %Relative efficiency of low-carbon technologies

%%Final Goods Production%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 1;                      %Normalize population
alpha = 0.3;                %Capital output share
v = 0.04;                   %Energy output share
Y2009 = 70000;              %Base year annual GDP in billions of USD
r2009 = 0.05;               %Base year annual net rate of return 
r2009d = ((1+r2009)^10)-1;  %Base yer decadal net rate of return

%%%Depreciation OPTION 1: delta = 100%
delta = 1;                              %Annual depreciation rate
Delta = (1-(1-delta)^10);               %Decadal depreciation rate
K0 = (alpha*Y2009*10)/(r2009d+Delta);   %Base year capital stock in billions of USD

% %%%Depreciation OPTION 2: delta = 65%, no recalibration:
% delta = 0.1;                            %Annual depreciation rate
% Delta = (1-(1-delta)^10);               %Decadal depreciation rate
% Delta1 = 1;                             %Decadal 100% depreciation rate
% K0 = (alpha*Y2009*10)/(r2009d+Delta1);  %Base year capital stock in billions of USD
 
% %Depreciation OPTION 3: delta = 65%, with recalibration:
% delta = 0.1;                            %Annual depreciation rate
% Delta = (1-(1-delta)^10);               %Decadal depreciation rate
% K0 = (alpha*Y2009*10)/(r2009d+Delta);   %Base year capital stock in billions of USD
 
 pi00 = 1;               %Base period share of labor devoted to final goods production
 E1_2008 = 3.43+1.68;    %GtC per annum
 E2_2008 = 3.75;         %GtC per annum
 E3_2008 = 1.95;         %GtC-eq per annum
 E0_2008 = ((kappa1*E1_2008^rho)+(kappa2*E2_2008^rho)+(kappa3*E3_2008^rho))^(1/rho);
 E0 = E0_2008*10;        %GtC per decade
 A0 = (Y2009*10)/((exp((-gamma(1))*((S1_2000+S2_2000)-Sbar)))*((K0^alpha)*((N*pi00)^(1-alpha-v))*(E0^v)));  %Initial TFP based on Decadal production function


%%%Productivity Growth Rates%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Energy Sector%%%
gZa_en = 0.02;                                         %Annual labor productivity growth rate (energy sectors)
gZ_en = ((1+gZa_en)^10)-1;                             %Decadal labor productivity growth rate (energy sectors)

%%%Final Goods Sector OPTION 1: Specify Labor Productivity Growth%%%
%           gZa_y = 0.02;                               %Annual labor productivity growth rate in final goods sector
%           gAa_y = (1+gZa_y)^(1-alpha-v);              %Corresponding TFP growth
%           gZd_y = ones(T+n,1)*(((1+gZa_y)^10)-1);     %Decadal labor productivity growth rate (all sectors)
%  
%%%Final Goods Sector OPTION 2: Specify TFP Growth%%%
%            gAa_y = 0.02;                            %Annual TFP growth rate (final output sector)
             gAa_y = 0;                               %Alt. Annual TFP growth rate (final output sector)
             gZa_y = ((1+gAa_y)^(1/(1-alpha-v)))-1;   %Corresponding annual labor productivity growth rate (final output sector)
             gAd_y = ((1+gAa_y)^10)-1;                %Decadal TFP growth rate (final output sector)
             gZd_y = ones(T+n,1)*(((1+gZa_y)^10)-1);  %Decadal labor productivity growth rate (final output sector)
 
%%%Final Goods Sector OPTION 3: DICE Model TFP Growth%%%
%     gANH0 = 0.160023196685654;                   %Initial decade (2005-2015) TFP growth rate
%     gammaNH0 = 0.00942588385340332;              %Rate of decline in productivity growth rate (percent per year)
%     gammaNH1 = 0.00192375245926376;              %Rate of decline of decline in productivity growth rate (percent per year)
%     gANH_y = zeros(T,1);
%     for i = 1:1:T;
%              gANH_y(i) = gANH0*exp(((-gammaNH0)*10*(i))*exp((-gammaNH1)*10*(i)));
%     end
%     for j = 1:1:n,
%         gANH_y(T+j) = gANH_y(T);
%     end
%     gANHa = zeros(T+n,1);
%     gANHa_y = ((1+gANH_y(T))^(1/10))-1;         %Annual long-run TFP growth rate
%     gZd_y = zeros(T+n,1);                      %Decadal labor productivity growth rate
%     for i = 1:1:T+n,
%         gZd_y(i) = ((1+gANH_y(i))^(1/(1-alpha-v)))-1;
%         gANHa(i) = ((1+gANH_y(i))^(1/10))-1;
%     end
%     z = 35;
%     plot(y2(1:z),(gANHa(1:z)*100))
%     xlabel('Year','FontSize',11)
%     ylabel('gTFP in Percent per Year','FontSize',11)
%     title('2010-DICE Model Annual TFP Growth','FontSize',13)
  

%%Final Good Sector TFP Levels%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
At = zeros(T,1);
At(1) = A0;                 
for i = 1:1:T-1;
   At(i+1) = At(i)*(1+gZd_y(i))^(1-alpha-v);     
end

%%Long-run Output Growth Rate on BGP%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gZBGP = gZd_y(T);             
% gZBGP = gZ_en;      %Alternative possible value for gTFP=1.5% to roughly account for declining oil output   

%%Utility%%
%%%%%%%%%%%
% sigma = 0.5;      
 sigma = 1;         %Logarithmic preferences
% sigma = 1.5;
% sigma = 2 ;

%%Beta OPTION 1: Specify exogenously%%%
beta = (.985)^10;   %Decadal discount rate
% beta = (.999)^10;

%%Beta OPTION 2: Calibrate to maintain effective discount factor = .985%%%
% beta_hat = ((.985)^10)/((1+gZd_y(1))^(1-sigma))
% beta = beta_hat;
    
  
%%Coal production%%
%%%%%%%%%%%%%%%%%%%
A2t = zeros(T,1);
A2t(1) = 7693;          
for i = 1:1:T-1;
    A2t(i+1) = A2t(i)*(1+gZ_en);
end

%%Coal Emissions%%
%%%%%%%%%%%%%%%%%%
ypsilon = zeros(T,1);   %Coal carbon emissions coefficient
a_yps = 8;              %Logistic curve coefficient
b_yps = -0.05;          %Logistic curve coefficient
for i = 1:1:T+n;
     ypsilon(i) = 1/(1+exp((-1)*(a_yps+b_yps*(i-1)*10)));
end

%%Wind production%%
%%%%%%%%%%%%%%%%%%%
%A3t = zeros(T,1);
%A3t(1) = 1311;
%for i = 1:1:T-1;
    %A3t(i+1) = A3t(i)*(1+gZ_en);
%end


%%Initial Oil Stock%%
%%%%%%%%%%%%%%%%%%%%%
R0 = 253.8;     %GtC

%%%%%%%%%%%%           Self Added for Chazel replication          %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%!!!!!!!!! CHANGE LATER!!!!!!!!!!!!
M0 = 2000;    % Initial mineral endowment in megatonnes, Mt
Ms0 = 19;  % Initial secondary mineral stock, Mt

delta_G = 1;     % Depreciation rate of green capital

rho_E3 = -3; % Parameter of substitution E3
rho_green = 0.5; % Parameter of substitution Gt 
psi = 1.877; % Energy obtained from given amount of Gt, in Gt/MtCu 

kappaL = 0.25; % Relative efficiency of labour in the production of E3
kappaG = 1-kappaL; % Relative efficiency of green capital in production of E3
kappaP = 0.6915; % Share parameter primary minerals in the production of Gt
kappaS = 1-kappaP; % Share parameter of secondary minerals in the production of Gt 


%%Labour Productivities%%
%%%%%%%%%%%%%%%%%%%%%%%%%
A3t = zeros(T,1);
A3t(1) = 865.14; % Initial labour productivity in the low carbon energy sector E3, in Gt/L
for i = 1:1:T-1;
    A3t(i+1) = A3t(i)*(1+gZ_en); 
end

Ap = zeros(T,1);
Ap(1) = 132000; % Initial labour productivitty in the primary mineral sector, Mt/L 
for i = 1:1:T-1;
    Ap(i+1) = Ap(i)*(1+gZ_en);
end

As = zeros(T,1);
As(1) = 132000; 
for i = 1:1:T-1;
As(i+1) = As(i)*(1+gZ_en); % Initial Labour productivity in secondary mineral sector, Mt/L
end 

%%Graph for Figure S.1%%
figure;
%the original line -plot(y,ypsilon,'-o')- gave an x-axis until 3400 therefore changed to below 
plot(y,ypsilon(1:T),'-o')
xlabel('Year','FontSize',11)
ylabel('Coal Emissions Coefficient','FontSize',11)
title('Coal Emissions Coefficient','FontSize',13)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%      Section 2: Solve for Optimal Choice Variables X        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% OLD: vars = 2*T+2*(T-1);     %Number of variables
% NEW: 
vars = 4*T+2*(T-1); 

%% Where: 
% - 4*T = for labour share of coal, wind, primary mineral, and secondary mineral and 
% - 3*(T-1) = for savings rate, oil stock remaining and mineral stock remaining 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2, Step 1: Define upper and lower bounds %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lb = zeros(vars,1);
ub = ones(vars,1);
for i = 1:1:T-1;
    ub(i) = 1;              %For savings rate (1-29)
    lb(i) = 0.00001;        %For savings rate "
    ub((T-1)+i) = R0;       %For oil stock remaining Rt (30-58)
    lb((T-1)+i) = 0.00001;  %For oil stock remaining Rt "

end
for i = 1:1:4*T;
    ub(2*(T-1)+i) = 1;        %For labor shares of coal (59-88) wind (89-118) primary mineral (119-148) and secondary mineral (149-178)
    lb(2*(T-1)+i) = 0.001;  %For labour shares " 
end
%for i = 1:1:T-1;
    %ub((2*(T-1)+4*T)+i) = M0;          %For mineral stock remaining Mt (179-207)
    %lb((2*(T-1)+4*T)+i) = 0.00001;     %For mineral stock remaining "
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2, Step 2 :Make Initial Guess x0 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% OPTION 1: USE PREVIOUS RESULTS %%

%%Note: The best x0 can be found by loading the saved output below
%%for the scenario that corresponds most closly to the one being run, and
%%then setting x0 = x. All file names indicate the parameters assumed,
%%e.g.: 'x_sig1_g0_b985_d1' is the optimal allocation for sigma=1 (sig1), 
%%annual TFP growth of 0% (g0), an annual discount factor of beta=0.985
%%(b985), and a decadal depreciation rate of Delta=100% (d1).

%%Sigma=1%%
%COMMENTED OUT (TO LOAD PREVIOUS RESULT) 

%load('x_sig1_g0_b985_d1_chazel','x')

%COMMENTED OUT TO ENSURE X0 LOAD PREVIOUS RESULTS X
%x0 = x;

%COMMENTED IN 
%%% OPTION 2: NEUTRAL STARTING POINT %%

x0 = zeros(vars,1);
for i = 1:1:T;
     x0(2*(T-1)+i) = 0.002;                             %coal labour share
     x0(2*(T-1)+T+i) = 0.01;                            %low carbon energy labour share
     x0(2*(T-1)+2*T+i) = 0.002;                                 %primary minerals labour share
     x0(2*(T-1)+3*T+i) = 0.002;                                 %secondary minerals labour share
end

for i = 1:1:T-1;
         %x0(2*(T-1)+4*T+i) = M0-((M0/1.1)/T)*i;           %initial primary mineral stock
         x0(i) = 0.25;                                      %savings rate
         x0((T-1)+i) = R0-((R0/1.1)/T)*i;                   %oil stock remaining
 end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Section 2, Step 3: Check Constraints and Objective Function Value at x0%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%f = CHAZEL_Objective(x0,A2t,A3t,At,Delta,K0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,phi,phi0,phiL,rho,sigma,v,ypsilon)
%[c, ceq] = CHAZEL_Constraints(x0,A2t,A3t,At,Delta,K0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,phi,phi0,phiL,rho,sigma,v,ypsilon)

f = CHAZEL_Objective(x0,A2t,A3t,Ap,As,At,Delta,delta_G,K0,M0,Ms0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,kappaG,kappaL,kappaP,kappaS,phi,phi0,phiL,psi,rho,rho_E3,rho_green,sigma,v,ypsilon)
[c, ceq] = CHAZEL_Constraints(x0,A2t,A3t,Ap,As,At,Delta,delta_G,K0,M0,Ms0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,kappaG,kappaL,kappaP,kappaS,phi,phi0,phiL,psi,rho,rho_E3,rho_green,sigma,v,ypsilon)

%%CHECK
length(x0)
length(ub)

%%%%%%%%%%%
%%%SOLVE%%%
%%%%%%%%%%%
disp('Starting fmincon optimization...');
options = optimoptions(@fmincon,'Tolfun',1e-12,'TolCon',1e-12,'MaxFunEvals',500000,'MaxIter',6200,'Display','iter','MaxSQPIter',10000,'Algorithm','active-set');
%[x, fval,exitflag] = fmincon(@(x)CHAZEL_Objective(x,A2t,A3t,At,Delta,K0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,phi,phi0,phiL,rho,sigma,v,ypsilon), x0, [], [], [], [], lb, ub, @(x)CHAZEL_Constraints(x,A2t,A3t,At,Delta,K0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,phi,phi0,phiL,rho,sigma,v,ypsilon), options);
[x, fval,exitflag] = fmincon(@(x)CHAZEL_Objective(x,A2t,A3t,Ap,As,At,Delta,delta_G,K0,M0,Ms0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,kappaG,kappaL,kappaP,kappaS,phi,phi0,phiL,psi,rho,rho_E3,rho_green,sigma,v,ypsilon), x0, [], [], [], [], lb, ub, @(x)CHAZEL_Constraints(x,A2t,A3t,Ap,As,At,Delta,delta_G,K0,M0,Ms0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,kappaG,kappaL,kappaP,kappaS,phi,phi0,phiL,psi,rho,rho_E3,rho_green,sigma,v,ypsilon), options);
%[x, fval,exitflag] = fmincon(@(x)CHAZEL_Objective(x,A2t,A3t,Ap,As,At,Delta,delta_G,K0,M0,Ms0,N,R0,S1_2000,S2_2000,Sbar,T,alpha,beta,gZ_en,gZd_y,gZBGP,gamma,kappa1,kappa2,kappa3,kappaG,kappaL,kappaP,kappaS,phi,phi0,phiL,psi,rho,rho_E3,rho_green,sigma,v,ypsilon), x0, [], [], [], [], lb, ub,[], options);
disp('fmincon finished!');
disp(['Exitflag: ', num2str(exitflag)]);

%%Save Output%%
%%%%%%%%%%%%%%%
%File name structure:
%Version#_sigma_gTFP_beta_delta_notes

save('x_sig1_g0_b985_d1_chazel','x')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%      Section 3: Compute Allocations and Carbon Taxes  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%
%%Energy%%
%%%%%%%%%%
oil = zeros(T,1);
    oil(1) = R0-x(T);
for i = 1:1:T-2;
    oil(1+i) = x(T+i-1)-x(T+i);
end
    ex_Oil = (x(T-1+T-2)-x(T-1+T-1))/(x(T-1+T-2));    %Fraction of oil left extracted in period T-1
    oil(T) = x(T-1+T-1)*ex_Oil;
ex_rates = zeros(T-1,1);
for i = 1:1:T-1;
    ex_rates(i) = oil(i)/x(T+i-1);
end
coal = zeros(T,1);
for i = 1:1:T;
    coal(i) = x(2*(T-1)+i)*A2t(i)*N;
end
%wind = zeros(T,1);
%for i = 1:1:T;
    %wind(i) = x(2*(T-1)+T+i)*(A3t(i)*N);
%end

%%%%%%%% SELF ADDED %%%%%%%%%%%%%%%%%
%%Primary mineral extraction, eq(12)
min_p = zeros(T,1);
for i = 1:1:T;
    min_p(i) = x(2*(T-1)+2*T+i)*(Ap(i)*N);
end

%%Secondary mineral extraction, eq(15)
min_s = zeros(T,1);
for i = 1:1:T;
    min_s(i) = x(2*(T-1)+3*T+i)*(As(i)*N);
end

%%Green capital production, eq (11) 
green = zeros(T,1);
for i = 1:1:T;
    green(i) = ((kappaS*min_s(i)^rho_green)+(kappaP*min_p(i)^rho_green))^(1/(rho_green));
end 

%% Primary mineral stock 
Mp = zeros(T,1);
Mp(1) = M0;
for i = 1:1:T-2;
    Mp(1+i) = Mp(i) - min_p(i);
end
    ex_Mp = (Mp(28)-Mp(29))/(Mp(28));    %Fraction of Mp left extracted in final period
    Mp(T) = Mp(29)*ex_Mp;

%% Secondary mineral stock
Ms = zeros(T,1);
Ms(1) = Ms0; % Stock at end of period 1
for i = 1:1:T-2;
    Ms(i+1) = Ms(i) + min_p(i);
end
    ex_Ms = (Ms(28)-Ms(29))/(Ms(28));    %Fraction of Ms left extracted in final period
    Ms(T) = Ms(29)*ex_Ms;

%%Low carbon energy production, eq (10)
E3 = zeros(T,1);
for i = 1:1:T;
       E3(i) = (kappaL*(x(2*(T-1)+T+i)*A3t(i)*N)^rho_E3+kappaG*(psi*green(i))^rho_E3)^(1/rho_E3);
end

%%Same as GHKT and Chazel: 
energy = zeros(T,1);
for i = 1:1:T; 
    %energy(i) = ((kappa1*oil(i)^rho)+(kappa2*coal(i)^rho)+(kappa3*wind(i)^rho))^(1/rho);
    energy(i) = ((kappa1*oil(i)^rho)+(kappa2*coal(i)^rho)+(kappa3*E3(i)^rho))^(1/rho);
end

%% compute fossil fuel use
fossil_fuel = zeros(T,1);
for i = 1:1:T;
    fossil_fuel(i) = oil(i) + coal(i);
end

%%%%%%%%%%%%%
%%Emissions%%
%%%%%%%%%%%%%
emiss = zeros(T,1);
for i = 1:1:T;
    emiss(i) = oil(i)+ypsilon(i)*coal(i);
end
S1t = zeros(T,1);        %Non-depreciating stock
S2t_Sbar = zeros(T,1);   %Depreciating stock (S2t-Sbar)
St = zeros(T,1);         %Total carbon concentrations

S1t(1) = S1_2000+phiL*emiss(1);
S2t_Sbar(1) = (1-phi)*(S2_2000-Sbar)+phi0*(1-phiL)*emiss(1);
St(1) = Sbar+S1t(1)+S2t_Sbar(1);

for i = 1:1:T-1;
    S1t(1+i) = S1t(i)+phiL*emiss(1+i);
    S2t_Sbar(1+i) = (1-phi)*S2t_Sbar(i)+phi0*(1-phiL)*emiss(1+i);
    St(1+i) = Sbar+S1t(1+i)+S2t_Sbar(1+i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Output and Consumption through T%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Yt = zeros(T,1);
Ct = zeros(T,1);
Kt1 = zeros(T,1);
% Yt(1) = At(1)*(exp((-gamma(1))*(St(1)-Sbar)))*(K0^alpha)*((1-x(2*(T-1)+1)-x(2*(T-1)+T+1)*N)^(1-alpha-v))*(energy(1)^v);
Yt(1) = At(1)*(exp((-gamma(1))*(St(1)-Sbar)))*(K0^alpha)*(((1-x(2*(T-1)+1)-x(2*(T-1)+T+1)-x(2*(T-1)+2*T+1)-x(2*(T-1)+3*T+1))*N)^(1-alpha-v))*(energy(1)^v);
Ct(1) = (1-x(1))*Yt(1);
Kt1(1) = x(1)*Yt(1)+(1-Delta)*K0;
for i = 1:1:T-2;
    %Yt(1+i) = At(1+i)*(exp((-gamma(1+i))*(St(1+i)-Sbar)))*(Kt1(i)^alpha)*(((1-x(2*(T-1)+1+i)-x(2*(T-1)+T+1+i))*N)^(1-alpha-v))*(energy(1+i)^v);
    Yt(1+i) = At(1+i)*(exp((-gamma(1+i))*(St(1+i)-Sbar)))*(Kt1(i)^alpha)*(((1-x(2*(T-1)+1+i)-x(2*(T-1)+T+1+i)-x(2*(T-1)+2*T+i)-x(2*(T-1)+3*T+i))*N)^(1-alpha-v))*(energy(1+i)^v);
    Kt1(1+i) = x(1+i)*Yt(1+i)+(1-Delta)*Kt1(i);
    Ct(1+i) = (1-x(i+1))*Yt(1+i); 
end
%Yt(T) = At(T)*(exp((-gamma(T))*(St(T)-Sbar)))*(Kt1(T-1)^alpha)*(((1-x(2*(T-1)+T)-x(2*(T-1)+2*T))*N)^(1-alpha-v))*(energy(T)^v);
Yt(T) = At(T)*(exp((-gamma(T))*(St(T)-Sbar)))*(Kt1(T-1)^alpha)*(((1-x(2*(T-1)+T)-x(2*(T-1)+2*T)-x(2*(T-1)+3*T)-x(2*(T-1)+4*T))*N)^(1-alpha-v))*(energy(T)^v);
theta = x(T-1);  %Savings rate
Ct(T) = Yt(T)*(1-theta);
Kt1(T) = theta*Yt(T)+(1-Delta)*Kt1(T-1);


%Compare savings rate theta to predicted BGP savings rate:
%theta_BGP = alpha*(((((1+gZBGP)^sigma)/beta)-(1-Delta))^(-1))*(1+gZBGP-1+Delta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Output and Consumption past T to T+n%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ktn = zeros(n+1,1);
Ytn = zeros(n,1);
Ktn(1) = Kt1(T); 
oiln = zeros(n,1);
En = zeros(n,1);

for i = 1:1:n;
    At(T+i) = At(T+i-1)*(1+gZd_y(T+i-1))^(1-alpha-v);
    oiln(i) = ex_Oil*x(2*(T-1))*((1-ex_Oil)^i);
    %En(i) = ((kappa1*oiln(i)^rho)+(kappa2*(coal(T)*(1+gZ_en)^i)^rho)+(kappa3*(wind(T)*(1+gZ_en)^i)^rho))^(1/rho);
    En(i) = ((kappa1*oiln(i)^rho)+(kappa2*(coal(T)*(1+gZ_en)^i)^rho)+(kappa3*(E3(T)*(1+gZ_en)^i)^rho))^(1/rho);
    %Ytn(i) = At(T+i)*(exp((-gamma(T))*(St(T)-Sbar)))*(Ktn(i)^alpha)*(((1-x(2*(T-1)+2*T)-x(2*(T-1)+T))*N)^(1-alpha-v))*(En(i)^v);
    Ytn(i) = At(T+i)*(exp((-gamma(T))*(St(T)-Sbar)))*(Ktn(i)^alpha)*(((1-x(2*(T-1)+T)-x(2*(T-1)+2*T)-x(2*(T-1)+3*T)-x(2*(T-1)+4*T))*N)^(1-alpha-v))*(En(i)^v);
    Ct(T+i) = (1-theta)*Ytn(i);
    Ktn(i+1) = theta*Ytn(i)+(1-Delta)*Ktn(i);
    Yt(T+i) = Ytn(i);
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%Optimal Carbon Taxes%%
%%%%%%%%%%%%%%%%%%%%%%%%

%%Goal: Plug allocations into optimal tax formula (paper equation (9))%%

%%Step 1: Compute vectors of marginal utilities and marginal emissions impacts {dSt+j/dEt}%%
MU = zeros(T+n,1);        %Marginal utility
MD = zeros(T+n,1);        %Marginal emissions impact on St {dSt+j/dEt}
for i = 1:1:T+n;
    MU(i) = Ct(i)^(-sigma);
    MD(i) = phiL+(1-phiL)*phi0*(1-phi)^(i-1);
end

%%Step 2: Compute Tax Path%%%
carbon_tax = zeros(T,1);    %Carbon tax level in $/mtC [since Yt is in $ billions and Et is in GtC]
lambda_hat = zeros(T,1);    %Carbon tax/GDP ratio

for i = 1:1:T+n;
    temp2 = zeros(T+n-i+1,1);
        for j = 1:1:T+n-i+1;
            temp2(j) = (beta^(j-1))*(MU(i+j-1)/MU(i))*(-gamma(T))*Yt(i+j-1)*MD(j);
        end
     carbon_tax(i) = sum(temp2)*(-1);
     lambda_hat(i) = carbon_tax(i)/Yt(i);
end

%%Diagnostic plot preview:
 z = 30;
 plot(y2(1:z),lambda_hat(1:z));
 xlabel('Year','FontSize',11);
 ylabel('Carbon Tax/GDP','FontSize',11);
 ylim([3.5e-05, 8.5e-05]);
 title('Carbon Tax/GDP','FontSize',13);
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%      Section 4: Save Allocations and Carbon Taxes  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Note: Only save for appropriate model scenario

energy_chazel_v1 = energy;
save('energy_chazel_v1','energy_chazel_v1')
fossil_fuel_chazel_v1 = fossil_fuel;
save('fossil_fuel_chazel_v1','fossil_fuel_chazel_v1')
oil_chazel_v1 = oil;
save('oil_chazel_v1','oil_chazel_v1')
ex_rates_chazel_v1 = ex_rates
save('ex_rates_chazel_v1','ex_rates_chazel_v1')
coal_chazel_v1 = coal;
save('coal_chazel_v1','coal_chazel_v1')
E3_chazel_v1 = E3;
save('E3_chazel_v1','E3_chazel_v1')
lambda_hat_chazel_v1 = lambda_hat;
save('lambda_hat_chazel_v1','lambda_hat_chazel_v1')
carbon_tax_chazel_v1 = carbon_tax;
save('carbon_tax_chazel_v1','carbon_tax_chazel_v1')
Yt_chazel_v1 = Yt;
save('Yt_chazel_v1','Yt_chazel_v1')
Ct_chazel_v1 = Ct;
save('Ct_chazel_v1','Ct_chazel_v1')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%      Section 5: Graph Optimal Carbon Taxes     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%Graph Carbon Tax-GDP Ratio%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('lambda_hat_chazel_v1','lambda_hat_chazel_v1')

z = 30;
figure;
plot(y2(1:z), lambda_hat_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('Carbon Tax/GDP', 'FontSize', 11);
ylim([7.5e-05, 30.5e-05]);
title('Carbon Tax to GDP ratio (chazel replication)');


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%Graph Carbon Tax Level%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('carbon_tax_chazel_v1','carbon_tax_chazel_v1')

z = 10;
figure;
plot(y2(1:z), carbon_tax_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('Carbon Tax ($/mtC)', 'FontSize', 11);
title('Carbon Tax (chazel replication)');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%  Energy Use Over Time  %%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Energy
load('energy_chazel_v1.mat','energy_chazel_v1')

z = 30;
figure;
plot(y2(1:z), energy_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('GtC', 'FontSize', 11);
title('Energy Use (chazel replication)');

%% Fossil Fuel
load('fossil_fuel_chazel_v1.mat','fossil_fuel_chazel_v1')

z = 30;
figure;
plot(y2(1:z), fossil_fuel_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('GtC', 'FontSize', 11);
title('Fossil Fuel Use (chazel replication)');

%% Oil
load('oil_chazel_v1.mat','oil_chazel_v1')

z = 30;
figure;
plot(y2(1:z), oil_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('Oil Use', 'FontSize', 11);
title('Oil Use (chazel replication)');

%% Fraction of oil left extracted
load('ex_rates_chazel_v1.mat','ex_rates_chazel_v1')

z = 29;
figure;
plot(y2(1:z), ex_rates_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('Rate', 'FontSize', 11);
title('Extraction rates of oil (chazel replication)');

%% Coal
load('coal_chazel_v1.mat','coal_chazel_v1')

z = 30;
figure;
plot(y2(1:z), coal_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('Coal Use', 'FontSize', 11);
title('Coal Use (chazel replication)');

%% Wind
load('E3_chazel_v1.mat','E3_chazel_v1')

z = 30;
figure;
plot(y2(1:z), E3_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('Low Carbon Energy', 'FontSize', 11);
title('Low Carbon Energy (chazel replication)');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%  GDP Growth Over Time  %%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('Yt_chazel_v1.mat','Yt_chazel_v1')

z = 10;
figure;
plot(y2(1:z), Yt_chazel_v1(1:z), ' -b', 'LineWidth', 1.5);
xlabel('Year', 'FontSize', 11);
ylabel('Output', 'FontSize', 11);
title('GDP (chazel replication)');