using LinearAlgebra
using SparseArrays
using Printf
using Random
using Plots
using Statistics

"""
    FPCA算法实现

参数:
    U0: 初始正交矩阵 (p×r)
    Rk_list: 对称正定矩阵列表 [R1, R2, ..., RK]
    ϵ: 收敛阈值 (默认 1e-5)
    max_iters: 最大迭代次数 (默认 100)
    verbose: 是否打印详细输出 (默认 true)

返回:
    U_opt: 最优正交矩阵
    history: 包含迭代历史的字典
"""
function FPCA_algorithm(U0, Rk_list, ϵ=1e-7, max_iters=100, verbose=true)
    # 记录开始时间
    start_time = time()
    
    # 获取维度信息
    p, r = size(U0)
    K = length(Rk_list)
    
    # 初始化
    Ut = copy(U0)
    t = 0
    
    # 存储历史记录
    history = Dict(
        :f_values => Float64[],
        :mu_values => Vector{Float64}[],
        :U_norm => Float64[],
        :iterations => 0,
        :times => Dict{String,Float64}()  # 添加计时记录
    )
    
    # 目标函数: f(U) = min_k tr(U^T R_k U)
    function objective_function(U)
        min_val = Inf
        for k in 1:K
            val = tr(U' * Rk_list[k] * U)
            if val < min_val
                min_val = val
            end
        end
        return min_val
    end
    
    # 计算当前目标函数值
    f_prev = objective_function(Ut)
    push!(history[:f_values], f_prev)
    
    if verbose
        println("="^60)
        println("开始FPCA算法")
        println("维度: p = $p, r = $r, K = $K")
        println("初始目标值: f(U0) = $f_prev")
        println("="^60)
    end
    
    # 主迭代循环
    for iter in 1:max_iters
        iter_start_time = time()
        
        if verbose
            println("\n=== 迭代 $iter ===")
        end
        
        # 步骤1: 计算 {A_k, c_k} (公式22)
        step1_time = @elapsed begin
            Ak_list = []
            ck_list = []
            
            for k in 1:K
                A_k = Ut' * Rk_list[k]  # r×p 矩阵
                c_k = tr(Ut' * Rk_list[k] * Ut)
                
                push!(Ak_list, A_k)
                push!(ck_list, c_k)
            end
        end
        
        # 步骤2: 计算 µ* (求解公式29)
        step2_time = @elapsed begin
            μ_star = solve_dual_problem(Ak_list, ck_list, verbose)
            push!(history[:mu_values], copy(μ_star))
        end
        
        if verbose
            @printf("µ* = [%s]\n", join([@sprintf("%.4f", μ) for μ in μ_star], ", "))
            @printf("µ* 分布: min=%.4f, max=%.4f", 
                    minimum(μ_star), maximum(μ_star))
        end
        
        # 步骤3: 更新 U_{t+1} (公式30)
        step3_time = @elapsed begin
            Ut = update_U(Ut, Ak_list, μ_star)
        end
        
        # 计算新的目标函数值
        f_current = objective_function(Ut)
        push!(history[:f_values], f_current)
        
        # 检查正交性
        orth_error = norm(Ut' * Ut - I)
        push!(history[:U_norm], orth_error)
        
        # 计算相对变化
        rel_change = abs(f_current - f_prev) / (abs(f_prev))
        
        # 记录各步骤时间
        history[:times]["step1_iter$iter"] = step1_time
        history[:times]["step2_iter$iter"] = step2_time
        history[:times]["step3_iter$iter"] = step3_time
        
        if verbose
            @printf("目标值: %.6f → %.6f (变化: %.2e)\n", f_prev, f_current, f_current - f_prev)
            @printf("相对变化: %.2e, 正交误差: %.2e\n", rel_change, orth_error)
            @printf("各步骤耗时: 步骤1=%.3fs, 步骤2=%.3fs, 步骤3=%.3fs\n", 
                    step1_time, step2_time, step3_time)
            
            # 打印当前约束值
            constraint_values = [tr(Ut' * Rk * Ut) for Rk in Rk_list]
            @printf("约束值: [%s]\n", join([@sprintf("%.4f", v) for v in constraint_values], ", "))
        end
        
        # 检查收敛条件
        if rel_change ≤ ϵ
            if verbose
                println("✓ 收敛于迭代 $iter")
            end
            history[:iterations] = iter
            break
        end
        
        f_prev = f_current
        t += 1
        
        if iter == max_iters
            if verbose
                println("⚠ 达到最大迭代次数 ($max_iters)")
            end
            history[:iterations] = max_iters
        end
        
        # 记录迭代总时间
        iter_time = time() - iter_start_time
        history[:times]["iter$iter"] = iter_time
        
        if verbose
            @printf("迭代总时间: %.3fs\n", iter_time)
        end
    end
    
    # 记录总运行时间
    total_time = time() - start_time
    history[:times]["total"] = total_time
    
    return Ut, history
end

"""
    求解对偶问题 (公式29)
    
    min_μ 2||A(μ)||_∗ + ∑μ_k c_k
    s.t. μ ≥ 0, ∑μ_k = 1
"""
function solve_dual_problem(Ak_list, ck_list, verbose=false; 
                                  max_iters=10000, tol=1e-8)
    """
    简单的次梯度投影法，适用于小规模问题
    """
    K = length(Ak_list)
    r, p = size(Ak_list[1])
    
    # 初始化
    μ = ones(K) / K
    best_μ = copy(μ)
    
    # 计算初始目标值
    A_μ = zeros(p, r)
    for k in 1:K
        A_μ += μ[k] * Ak_list[k]'
    end
    best_obj = 2 * sum(svd(A_μ).S) + dot(ck_list, μ)
    
    # 步长参数
    α0 = 0.1
    best_iter = 0
    
    for iter in 1:max_iters
        # 1. 计算 A(μ)
        A_μ = zeros(p, r)
        for k in 1:K
            A_μ += μ[k] * Ak_list[k]'
        end
        
        # 2. 计算 SVD
        F = svd(A_μ)
        U, S, V = F.U, F.S, F.V
        
        # 3. 计算次梯度
        # 注意：当 S 中有 0 时，次梯度不唯一
        # 我们使用 UV' 作为次梯度（对应最大奇异值的方向）
        subgrad_nuclear = U * V'
        
        grad = zeros(K)
        for k in 1:K
            grad[k] = 2 * sum(Ak_list[k] .* subgrad_nuclear) + ck_list[k]
        end
        
        # 4. 更新步长（递减）
        α = α0 / sqrt(iter + 1)
        
        # 5. 次梯度更新
        μ_new = μ - α * grad
        
        # 6. 投影到单纯形
        μ_new = project_to_simplex(μ_new)
        
        # 7. 计算新目标值
        A_μ_new = zeros(p, r)
        for k in 1:K
            A_μ_new += μ_new[k] * Ak_list[k]'
        end
        obj_new = 2 * sum(svd(A_μ_new).S) + dot(ck_list, μ_new)
        
        # 8. 更新最佳解
        if obj_new < best_obj - 1e-10
            best_obj = obj_new
            best_μ = copy(μ_new)
            best_iter = iter
        end
        
        # 9. 检查收敛
        if iter > 100 && norm(μ_new - μ) < tol
            if verbose
                println("在迭代 $iter 收敛")
            end
            break
        end
        
        μ = μ_new
        
        if verbose && iter % 1000 == 0
            @printf("迭代 %6d: 目标值 = %.8f, 最佳 = %.8f, 步长 = %.6f\n", 
                   iter, obj_new, best_obj, α)
        end
    end
    
    if verbose
        println("最终最佳目标值: ", best_obj)
        println("在迭代 ", best_iter, " 获得最佳解")
    end
    
    return best_μ
end

function solve_dual_problem(Ak_list, ck_list; 
                                 max_iters=100, 
                                 tol=1e-6,
                                 verbose=true)
    """
    交替最小化算法求解：
    min_μ 2||A(μ)||_* + ∑μ_k c_k
    s.t. μ ≥ 0, ∑μ_k = 1
    
    其中 A(μ) = ∑ μ_k A_k^T
    """
    
    K = length(Ak_list)
    r, p = size(Ak_list[1])  # A_k: r × p
    
    # 初始化 μ (均匀分布)
    μ = ones(K) / K
    μ_prev = zeros(K)
    
    # 存储历史目标值
    f_values = Float64[]
    
    # 初始化 Φ
    A_μ = compute_Aμ(Ak_list, μ)
    Φ = compute_optimal_Phi(A_μ)
    
    iter = 0
    converged = false
    
    if verbose
        println("开始交替最小化算法...")
        println("K = $K, r = $r, p = $p")
    end
    
    while iter < max_iters && !converged
        # 1. 固定 μ，更新 Φ
        A_μ = compute_Aμ(Ak_list, μ)
        Φ = compute_optimal_Phi(A_μ)
        
        # 2. 固定 Φ，更新 μ (求解 QP)
        Q = compute_Q_matrix(Ak_list, Φ)
        
        # 使用 JuMP + Ipopt 求解 QP
        μ_new = solve_qp(Q, ck_list, verbose && iter==0)
        
        # 3. 计算目标值
        f_val = compute_objective(Ak_list, ck_list, μ_new, Φ)
        push!(f_values, f_val)
        
        # 4. 检查收敛
        if iter > 0
            rel_change = abs(f_values[end] - f_values[end-1]) / (abs(f_values[end-1]) + 1e-10)
            
            if verbose
                @printf("迭代 %3d: 目标值 = %.8e, 相对变化 = %.2e\n", 
                       iter, f_val, rel_change)
            end
            
            if rel_change < tol
                converged = true
                if verbose
                    println("在迭代 $iter 收敛")
                end
            end
        else
            if verbose
                @printf("迭代 %3d: 初始目标值 = %.8e\n", iter, f_val)
            end
        end
        
        μ_prev = copy(μ)
        μ = μ_new
        iter += 1
    end
    
    # 计算最终核范数
    A_μ_final = compute_Aμ(Ak_list, μ)
    nuclear_norm = sum(svd(A_μ_final).S)
    final_obj = 2 * nuclear_norm + dot(ck_list, μ)
    
    if verbose
        println("\n=== 算法结果 ===")
        println("总迭代次数: $iter")
        println("收敛: $converged")
        println("最终 μ: ", round.(μ, digits=6))
        println("∑μ = ", sum(μ))
        println("核范数: ", nuclear_norm)
        println("最终目标值: ", final_obj)
        println("QP目标值: ", f_values[end])
    end
    
    return μ#, f_values, converged
end

function compute_Aμ(Ak_list, μ)
    """计算 A(μ) = ∑ μ_k A_k^T"""
    K = length(Ak_list)
    r, p = size(Ak_list[1])
    
    A_μ = zeros(p, r)  # 注意转置
    for k in 1:K
        A_μ .+= μ[k] * Ak_list[k]'  # A_k^T
    end
    
    return A_μ
end

function compute_optimal_Phi(A_μ)
    """计算最优 Φ = (A(μ)^T A(μ))^{-1/2}"""
    # 计算 A_μ^T A_μ
    M = A_μ' * A_μ  # r × r 矩阵
    
    # 添加小正则项确保正定性
    ϵ = 1e-8
    M_reg = M + ϵ * I
    
    # 计算 M_reg^{-1/2}
    F = eigen(Symmetric(M_reg))
    Λ_inv_sqrt = Diagonal(1.0 ./ sqrt.(max.(F.values, 0.0) .+ 1e-12))
    Φ = F.vectors * Λ_inv_sqrt * F.vectors'
    
    return Φ
end

function compute_Q_matrix(Ak_list, Φ)
    """计算 Q 矩阵: Q[i,j] = Tr(A_i^T Φ A_j)"""
    K = length(Ak_list)
    Q = zeros(K, K)
    
    # 预计算 A_i^T Φ
    A_phi = [Ak_list[i]' * Φ for i in 1:K]  # 每个是 p × r
    
    for i in 1:K
        for j in i:K  # 利用对称性
            # Tr(A_i^T Φ A_j) = vec(A_i^T Φ)·vec(A_j)
            # 等价于 sum(A_phi[i] .* A_j)
            Q[i, j] = sum(A_phi[i] .* Ak_list[j]')
            if i != j
                Q[j, i] = Q[i, j]  # 对称矩阵
            end
        end
    end
    
    return Q
end

function solve_qp(Q, c, verbose_first=false)
    """求解 QP: min_μ μ^T Q μ + c^T μ, s.t. μ ≥ 0, ∑μ = 1"""
    K = length(c)
    
    model = Model(Ipopt.Optimizer)
    
    if !verbose_first
        set_silent(model)
    else
        println("求解 QP，维度 K = $K")
    end
    
    # 变量
    @variable(model, μ[1:K] >= 0)
    
    # 约束
    @constraint(model, sum(μ) == 1)
    
    # 目标函数
    @objective(model, Min, μ' * Q * μ + dot(c, μ))
    
    # 求解
    optimize!(model)
    
    # 检查求解状态
    if termination_status(model) != MOI.OPTIMAL
        @warn "QP 求解器未达到最优解: $(termination_status(model))"
    end
    
    μ_val = value.(μ)
    
    return μ_val
end

function compute_objective(Ak_list, ck_list, μ, Φ)
    """计算目标值: Tr(Φ⁻¹) + Tr(A(μ)^T A(μ)Φ) + ∑μ_k c_k"""
    
    # 1. 计算 A(μ)
    A_μ = compute_Aμ(Ak_list, μ)
    
    # 2. 计算 Tr(Φ⁻¹)
    F = eigen(Symmetric(Φ))
    Φ_inv = F.vectors * Diagonal(1.0 ./ max.(F.values, 1e-12)) * F.vectors'
    tr_phi_inv = tr(Φ_inv)
    
    # 3. 计算 Tr(A(μ)^T A(μ)Φ)
    tr_A_phi = tr(A_μ' * A_μ * Φ)
    
    # 4. 线性项
    linear_term = dot(ck_list, μ)
    
    return tr_phi_inv + tr_A_phi + linear_term
end

function compute_true_objective(Ak_list, ck_list, μ)
    """计算真实目标值: 2||A(μ)||_* + ∑μ_k c_k"""
    A_μ = compute_Aμ(Ak_list, μ)
    nuclear_norm = sum(svd(A_μ).S)
    return 2 * nuclear_norm + dot(ck_list, μ)
end
       
#function solve_dual_problem(Ak_list, ck_list, verbose=false; 
#                            max_inner_iters=1000, inner_ϵ=1e-6)
#   start_time = time()
#    K = length(Ak_list)
#    r, p = size(Ak_list[1])
    
    # 初始化 μ (均匀分布)
#    μ = ones(K) / K
#    μ_prev = copy(μ)
    
    # 对偶目标函数
#    function dual_objective(μ_vec)
        # 计算A(μ) = ∑ μ_k A_k^T
#        A_μ = zeros(p, r)
#        for k in 1:K
#            A_μ += μ_vec[k] * Ak_list[k]'
#        end
        
        # 计算核范数（奇异值之和）
    
#       svd_vals = svd(A_μ)
#        nuclear_norm = sum(svd_vals.S)
        
        # 计算加权和
#        weighted_sum = sum(μ_vec[k] * ck_list[k] for k in 1:K)
        
        # 返回目标函数值
#        return 2 * nuclear_norm + weighted_sum
#   end
    
    # 使用投影梯度下降求解
#    learning_rate = 0.1
#    best_μ = copy(μ)
#    best_obj = dual_objective(μ)
    
#    for inner_iter in 1:max_inner_iters
        # 计算梯度
        # 使用有限差分法计算梯度
    #    grad = zeros(K)
    #    ε = 1e-6
        
    #   for k in 1:K
            # 正向扰动
    #    μ_plus = copy(μ)
    #        μ_plus[k] += ε
    #        μ_plus = project_to_simplex(μ_plus)
            
            # 负向扰动
    #       μ_minus = copy(μ)
    #       μ_minus[k] -= ε
    #       μ_minus = project_to_simplex(μ_minus)
            
            # 计算数值梯度
    #       obj_plus = dual_objective(μ_plus)
    #       obj_minus = dual_objective(μ_minus)
            
    #       grad[k] = (obj_plus - obj_minus) / (2ε)
    #   end
        
        # 梯度下降更新
    #   μ_new = μ - learning_rate * grad
        
        # 投影到单纯形 (μ ≥ 0, ∑μ_k = 1)
    #    μ_new = project_to_simplex(μ_new)
        
        # 计算目标函数值
    #    obj_new = dual_objective(μ_new)
        
        # 检查是否改进
    #   if obj_new < best_obj
    #       best_obj = obj_new
    #       best_μ = copy(μ_new)
    # end
        
        # 检查内层收敛
    #   if norm(μ_new - μ) < inner_ϵ
    #       break
    #   end
        
    #   μ = μ_new
        
        # 自适应调整学习率
    #   if inner_iter % 50 == 0
    #       learning_rate *= 0.95
    #   end
        
        # 打印进度
    #   if verbose && inner_iter % 100 == 0
    #       @printf("内层迭代 %d: 目标值 = %.6f\n", inner_iter, obj_new)
    #   end
    #end
    
    #total_time = time() - start_time
    #if verbose
    #   @printf("对偶问题求解时间: %.3fs\n", total_time)
    #   @printf("最终目标值: %.6f\n", best_obj)
    #   @printf("最优μ: [%s]\n", join([@sprintf("%.4f", μ) for μ in best_μ], ", "))
    #end
    
    #return best_μ
#end


"""
    投影到单纯形 (μ ≥ 0, ∑μ_k = 1)
"""
#function project_to_simplex(v)
#   u = sort(v, rev=true)
#   n = length(v)
#   ρ = maximum([j for j in 1:n if u[j] + (1 - sum(u[1:j])) / j > 0])
    
#   λ = (1 - sum(u[1:ρ])) / ρ
#   proj = max.(v .+ λ, 0)
    
#   return proj
#end

"""
    更新 U (公式30)
    
    U_{new} = argmax_U min_μ tr(U^T (∑μ_k R_k) U)
    s.t. U^T U = I
"""
function update_U(Ut, Rk_list, μ_star)
    start_time = time()
    p, r = size(Ut)
    K = length(Rk_list)
    
    # 步骤1: 计算 R_μ = ∑ μ_k^* R_k
    R_μ = zeros(r, p)
    for k in 1:K
        R_μ += μ_star[k] * Rk_list[k]
    end
    
    
    # 步骤2:更新U
    U_new = R_μ'*(R_μ*R_μ')^(-1/2)

    
    total_time = time() - start_time
    return U_new
end

"""
    验证最终解的质量
"""
function verify_solution(U_final, Rk_list)
    println("\n" * "="^60)
    println("验证最终解")
    println("="^60)
    
    K = length(Rk_list)
    constraint_values = zeros(K)
    
    # 计算所有约束值
    for k in 1:K
        constraint_values[k] = tr(U_final' * Rk_list[k] * U_final)
    end
    
    println("约束值:")
    for k in 1:K
        @printf("  tr(U^T R_%d U) = %.6f\n", k, constraint_values[k])
    end
    
    min_val = minimum(constraint_values)
    max_val = maximum(constraint_values)
    mean_val = sum(constraint_values)/length(constraint_values)
    
    println("\n统计信息:")
    @printf("  最小值: %.6f\n", min_val)
    @printf("  最大值: %.6f\n", max_val)
    @printf("  平均值: %.6f\n", mean_val)
   
    if min_val > 0
        ratio = max_val / min_val
        @printf("  最大值/最小值: %.4f\n", ratio)
        
        if ratio < 1.5
            println("  ✓ 解的质量很好: 所有约束值接近")
        elseif ratio < 2.0
            println("  ⚠ 解的质量一般")
        else
            println("  ✗ 解的质量较差: 约束值差异较大")
        end
    end
    
    # 检查正交性
    orth_error = norm(U_final' * U_final - I)
    @printf("\n正交性检查:\n")
    @printf("  ||U^T U - I||_F = %.2e\n", orth_error)
    
    if orth_error < 1e-6
        println("  ✓ 正交性良好")
    else
        println("  ⚠ 正交性有偏差")
    end
    
    return constraint_values, min_val, max_val
end

"""
    生成测试数据
"""
function generate_test_data(p, r, K; seed=42)
    Random.seed!(seed)
    
    # 生成随机初始正交矩阵 U0
    #U0 =ones(Float64, p)
    #U0=reshape(U0,p,1)
    #U0 =LinearAlgebra.normalize(U0 + 1e0 * randn(Float64, size(U0)))
    
    # 生成随机对称正定矩阵 R_k
    
    n = p
    k = K
    As = [zeros(Float64, n, n) for i in 1:k]
    
    for l = 1:k

        for i in 1:n, j in i + 1:n
            As[l][i, j] = n
            As[l][j, i] = As[l][i, j]
        end
        for i in 1:n
            As[l][i, i] = l+sum(abs(As[l][i, m]) for m = 1:n) - abs(As[l][i, i])
            # last term dispensable since As are intialized as null
        end
    end
    Rk_list = As

    # 生成具有更复杂结构的矩阵
    #Rk_list = [rand(p, p) for _ in 1:K]
    #Rk_list = [0.5*(R + R') for R in Rk_list]  # 确保对称性
    #Rk_list = [R + p*I for R in Rk_list]  # 确保正定性
    # 生成随机正交矩阵
    U0 = randn(p, r)
    Q, _ = qr(U0)
    U0 = Q[:, 1:r]
    # 打印数据信息
    println("测试数据生成完成:")
    println("  p = $p, r = $r, K = $K")
    
    return U0, Rk_list
end

"""
    绘制收敛曲线
"""
function plot_convergence(history)
    f_values = history[:f_values]
    iterations = length(f_values) - 1
    #f_values=history[:f_values][end]*ones(iterations+1)-f_values
    print(f_values)
end

"""
    运行完整测试
"""
function run_full_test(p=10, r=1, K=2; ϵ=1e-10, max_iters=100)
    println("\n" * "="^60)
    println("FPCA算法完整测试")
    println("="^60)
    
    # 生成测试数据
    U0, Rk_list = generate_test_data(p, r, K)
    #M,A,B=Credit()
    #A,B=quick_load_matrices()
    #U0 = randn(p, r)
    #Q, R = qr(U0)
    #U0 = Q[:, 1:r]
    #Rk_list=[A,B]

    # 运行FPCA算法
    U_final, history = FPCA_algorithm(U0, Rk_list, ϵ, max_iters, true)
    
    # 验证结果
    constraint_values, min_val, max_val = verify_solution(U_final, Rk_list)
    
    # 绘制收敛曲线
    plot_convergence(history)
    
    return U_final, history, constraint_values
end

"""
    主函数
"""
function main()
    println("FPCA算法实现")
    println("\n" * "="^60)
    
    # 运行测试
    U_final, history, constraint_values = run_full_test(200,1 ,200)
    
    # 打印最终总结
    println("\n" * "="^60)
    println("最终总结")
    println("="^60)
    @printf("最终目标值: %.6f\n", history[:f_values][end])
    @printf("迭代次数: %d\n", history[:iterations])
    @printf("总运行时间: %.3fs\n", history[:times]["total"])
    
    # 打印各步骤平均耗时
    if haskey(history[:times], "step1_iter1")
        step1_times = [history[:times]["step1_iter$i"] for i in 1:history[:iterations]]
        step2_times = [history[:times]["step2_iter$i"] for i in 1:history[:iterations]]
        step3_times = [history[:times]["step3_iter$i"] for i in 1:history[:iterations]]
        
        println("\n各步骤平均耗时:")
        @printf("步骤1 (计算A_k,c_k): %.3fs\n", mean(step1_times))
        @printf("步骤2 (求解对偶问题): %.3fs\n", mean(step2_times))
        @printf("步骤3 (更新U): %.3fs\n", mean(step3_times))
    end
    
    println("\n算法完成!")
    
    return U_final#, history
end

# 如果直接运行此文件，执行主函数
main()
