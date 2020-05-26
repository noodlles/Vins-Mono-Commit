#include "marginalization_factor.h"

void ResidualBlockInfo::Evaluate()
{
    // 残差resize
    residuals.resize(cost_function->num_residuals());

    // 取参数变量维度
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    // 遍历参数变量
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        // block_sizes[i]： 第i个参数变量的维度
        // 将第i个参数变量对应的雅克比子块 resize成 (行: 残差维度 , 列: 第i个参数变量的维度)
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        // 这里是取指针，所以在下面"cost_function->Evaluate"函数调用的时候，雅克比的数据直接写到jacobians[i].data()上
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    // 调用cost func 重载的函数， 计算残差、雅克比
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    // 如果使用核函数
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}

//添加残差块相关信息（优化变量，待边缘化变量）
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    // 将residual_block_info 推入 factors容器
    factors.emplace_back(residual_block_info);

    /// 三种情况的 "parameter_blocks" 内容
    /// 视觉: [Ti, Tj, Tbc , 特征点逆深度]
    /// IMU: [滑动窗口第0帧IMU位姿， 第0帧速度\ba\bg , 滑动窗口第1帧IMU位姿， 第1帧速度\ba\bg  ]
    /// 上一次marg得到的: [ 上一次marg剩下的参数变量  ]
    ///
    // 取参数变量(指针集合)地址集合
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    // 取约束项相关变量的维度数， 也就是
    // 对应上面的"parameter_blocks"内元素的维度
    // IMU: [滑动窗口第0帧IMU位姿， 第0帧速度\ba\bg , 滑动窗口第1帧IMU位姿， 第1帧速度\ba\bg  ] 的参数维度
    // class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>    ====》 对应的是 <7,9,7,9>
    // 视觉: [Ti, Tj, Tbc , 特征点逆深度] 的参数维度
    // class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> ====》      <7,7,7,1>
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    // 遍历参数变量
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        // 取某个变量的地址
        double *addr = parameter_blocks[i];
        // 取该变量的维度数
        int size = parameter_block_sizes[i];
        // 将变量地址和维度数打包，存到parameter_block_size映射, <优化变量内存地址,global Size>
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    // 遍历drop_set
    // 视觉: drop_set={0,3} ===> 表示需要marg参数为 [ Ti , 特征点逆深度 ]
    // IMU: drop_set={0,1} ===> 表示需要marg参数为 [滑动窗口第0帧IMU位姿， 第0帧速度\ba\bg ]
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        // 取待边缘化的优化变量（数组的起始地址）指针
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        // 在parameter_block_idx映射中，把对应地址的项置零，表示边缘化掉
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

//计算每个边缘化约束因子，对应的Jacobian，并更新parameter_block_data(对应参数变量的内存地址"0x...")
void MarginalizationInfo::preMarginalize()
{
    // 遍历需要边缘化的 视觉、IMU 因子
    for (auto it : factors)
    {
        // 计算 IMU或者视觉约束的 残差、雅克比
        it->Evaluate();

        // 取约束项相关变量的维度数， 也就是
        // 对应"parameter_blocks"内元素的维度
        ///IMU: [滑动窗口第0帧IMU位姿， 第0帧速度\ba\bg , 滑动窗口第1帧IMU位姿， 第1帧速度\ba\bg  ] 的参数维度
        // class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>    ====》 对应的是 <7,9,7,9>
        ///视觉: [Ti, Tj, Tbc , 特征点逆深度] 的参数维度
        // class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> ====》      <7,7,7,1>
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();

        // 遍历参数变量
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            // 取参数变量i的地址 (数组的第一个元素地址)
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            // 取参数变量i的维度 (数组元素个数)
            int size = block_sizes[i];
            // 在"parameter_block_data"根据地址查找，如果还没有对应的记录，也就是还没有为参数申请内存空间
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                // 创建数组 ： 开辟内存空间
                double *data = new double[size];
                // 将参数变量i的地址 (数组的第一个元素地址) 复制 到数组 ， 数组的每个元素都是这个地址
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);

                ///[注意] addr 与 地址 的区别:
                ///      addr是地址的long类型， 如果某个参数变量i地址为 "0x7fb44c266570" ，那么 addr=10328408
                // 这里是为了方便做映射的操作？
                // 增加记录 <addr, 参数变量i的地址 (数组的第一个元素地址)数组>
                // 举例 parameter_block_data中的某条映射为: <10328408 , data[size]={0x7fb44c266570}>
                parameter_block_data[addr] = data;
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for (auto it : p->sub_factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

//多线程构造先验项舒尔补AX=b的结构，计算Jacobian和残差
void MarginalizationInfo::marginalize()
{
    int pos = 0;
    // 遍历"parameter_block_idx"容器
    for (auto &it : parameter_block_idx)
    {
        // it = <参数变量内存地址,在parameter_block_size中的id>
        // 设置待边缘化的变量的idx(local_size)
        it.second = pos;
        // parameter_block_size 中某条记录: <优化变量内存地址, 变量的global Size (如7或者9) >
        pos += localSize(parameter_block_size[it.first]);
    }

    // m: 需要marg掉的变量的总维度
    m = pos;

    // 遍历“parameter_block_size”容器
    // 因为在"void MarginalizationInfo::addResidualBlockInfo()"，parameter_block_idx只对添加了需要边缘化的变量的映射 <待边缘化的优化变量内存地址,0>
    // 现在向"parameter_block_idx"添加要保留的变量
    for (const auto &it : parameter_block_size)
    {
        // it = <参数变量内存地址, 变量的global Size (如7或者9) >

        // 检查 it 所代表的参数变量是否是需要被marg的变量
        // 满足if条件，表示 it 所代表的参数变量 是需要保留的变量，
        // 并且在"parameter_block_idx"容器中还没有记录，这里添加记录
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            // 进入if
            // 表示it代表的变量是 需要保留的变量， 记录索引(local size)
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    // n: 需要保留的变量的总维度数
    n = pos - m;

    //ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    TicToc t_summing;
    // pos: 所有变量(边缘化+保留)的维度数
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();

    // 理解上直接看 单线程版本的
    /*
    // 遍历factors (视觉、IMU、上一次marg的结果)
    for (auto it : factors)
    {
        /// 遍历参数变量parameter_blocks
        /// 三种情况的 "parameter_blocks" 内容
        /// 1. 视觉: [Ti, Tj, Tbc , 特征点逆深度]
        /// 2. IMU: [滑动窗口第0帧IMU位姿， 第0帧速度\ba\bg , 滑动窗口第1帧IMU位姿， 第1帧速度\ba\bg  ]
        /// 3. 上一次marg得到的: [ 上一次marg剩下的参数变量 ]
        ///
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            // 取该参数变量i的 local_size 索引
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            // 取该参数变量i的 local_size 维度
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            // 从 it->jacobians 截取对应的部分
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            // 再次遍历"parameter_blocks" ， 只计算右上对角部分就好了， 剩下的是对称的， 因此 j=i开始
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                // 取参数变量j的 local_size 索引
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                // 取参数变量j的 local_size 维度
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                // 从 it->jacobians 截取参数变量j对应的雅克比部分
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                // 计算J^T*J
                // 对角线上的
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    // 非对角线上的
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            // 计算b=J^T r
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread


    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; i--)
    {
        pthread_join( tids[i], NULL );
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());

    // debug by qpc
    Eigen::MatrixXd A_tmp_to_print=A;
    Eigen::VectorXd b_tmp_to_print=b;

    //TODO
    // 这是为了保持对称
    // 在数值计算的时候，难以保证对称，这里处理一下
    ///Amm: J^TJ 的左上角部分，需要被marg的部分
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    // 特征值分解求逆
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    // 舒尔补
    // bmm: 与Amm对应的 b部分， 也就是需要被marg的部分
    Eigen::VectorXd bmm = b.segment(0, m);
    // Amr: J^T J右上角部分
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    // Arm: J^T J左下角部分
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    // Arr: J^T J右下角部分
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    // brr: 与Arr对应的 b部分， 也就是需要保留的部分
    Eigen::VectorXd brr = b.segment(m, n);
    // 进行舒尔补
    // Arr: nxn , Arm : nxm  , Amm : mxm , Amr: mxn
    // A : nxn
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;  // b: n

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    // 相当于运算符
    // (如果 saes2.eigenvalues().array() > eps) , 那么值就取 saes2.eigenvalues().array()
    //  如果不满足， 那么，对应的元素直接取 0
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    // S: 矩阵A特征值
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    /** ***************************************************************************
     * @brief marg先验 ====> ceres的优化约束项
     * 由于使用ceres，对于marg得到的先验 H 和 b ，没法直接加到最终的系统正规方程中
     * 因此，这里把 marg得到的先验 转化为 ceres可以使用的 约束项，直接加到目标函数中
     * 具体转换过程如下：
     * @arg H = J^T J = 特征值分解 = V diag{sqrt(eig)} diag{sqrt(eig)} V^T
     * @arg J = diag{sqrt(eig)} V^T  <====>  J^T = V diag{sqrt(eig)}
     * @arg (J^T)^{-1} = diag{sqrt(eig)}^{-1} V^{-1} = diag{sqrt(eig)}^{-1} V^{T}
     *
     * @brief 由于先验信息为 H dx = b ===> J^T J dx = (-)J^T r
     *        接下来，由于不知道残差 r， 因此，使用上面的等式{b=(-)J^T r}来得到r的表达
     *        即: r = (J^T)^{-1} b = diag{sqrt(eig)}^{-1} V^{T} b
     * 最终，得到了先验信息转换成约束的 J和r
     * J = diag{sqrt(eig)} V^T
     * r = (J^T)^{-1} b = diag{sqrt(eig)}^{-1} V^{T} b
     * ****************************************************************************/

    // saes2.eigenvectors().transpose(): nxn

    // marg先验 ====> ceres的优化约束项
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());

    /** ******************************************************
     * debug
     * 打印输出:
     * 1. 整个marg的矩阵A，b
     * 2. marg之后的矩阵A，b
     * *******************************************************/
//    std::cout<<"=======================Debug START============================"<<std::endl;
//    std::ofstream debugFile;
//    static bool once=true;
//    if(!once)
//        return;
//    debugFile.open("/home/autoware/shared_dir/catkin_ws/DataOutput/marg_Total_A.csv", std::ios::out | std::ios::trunc);
//    if (debugFile.is_open()) {
//        for (int row = 0; row < A_tmp_to_print.rows(); row++) {
//            for (int column = 0; column < A_tmp_to_print.cols(); column++) {
//                if (column != A_tmp_to_print.cols() - 1) {
//                    debugFile << A_tmp_to_print(row,column) << ",";
//                }
//                else {
//                    debugFile << A_tmp_to_print(row,column);
//                }
//            }
//            debugFile <<",,"<<b_tmp_to_print(row)<<std::endl;
//        }
//    }
//    debugFile.close();
//    once=false;
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    // it : <参数变量内存地址, 基于local size 的参数变量索引(如 0,6,15,...)>
    for (const auto &it : parameter_block_idx)
    {
        // 对需要保留的变量进行操作
        if (it.second >= m)
        {
            /// it.first: 需要保留的变量的地址(long)

            // parameter_block_size[it.first]: 要保留的变量的global Size (如7或者9)
            keep_block_size.push_back(parameter_block_size[it.first]);
            // parameter_block_idx[it.first] : 要保留的变量的 基于local size 的参数变量索引(如 0+m,6+m,15+m,...)
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            // parameter_block_data[it.first]: 要保留的变量i的地址(变量i的第一个元素地址)数组，这些地址指向线性化点处的值
            keep_block_data.push_back(parameter_block_data[it.first]);

            // addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            // addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            // addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            // addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            // addr_shift <第i帧的位姿参数地址(long) , 第i-1帧的位姿参数真正地址>
            // 所以使用addr_shift 检索 第i帧位姿参数地址(long)，最终得到的是第i-1帧的位姿参数真正地址
            // 如果it.first = 原本第1帧的变量地址para_Pose[1](long)， 那么 push进去的是para_Pose[0] （也就是指向para_Pose第[0]个元素的指针）
            /// 这是因为，在下一次优化的时候，原本排在第i帧的数据已经变成了第i-1帧的数据了，
            /// （也就是说，原本 滑窗中的第1帧的变量是要保留的，但是在下一次优化的时候，这些数据排在了第0帧，这里提前把要保留的变量地址指向了第0帧）
            /// 所以，这里提前把地址指向了前一帧的数据地址，但是还没有对para_Pose[]这些参数变量进行重新赋值，
            /// 在slidewindo()函数中，会对PS[] RS[] ...等参数变量重新赋值，到那个时候，指针和数据才算完全对应上
            // 将要保留的变量地址(指向para_Pose[]..内元素的指针)保存到“last_marginalization_parameter_blocks”
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    // 求和，获得需要保留的变量的global Size的总和
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

// 根据上一次边缘化留下来的信息，构造新的边缘化因子
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    // 遍历std::vector<int> keep_block_size;中的元素
    for (auto it : marginalization_info->keep_block_size)
    {
        // 记录到 父类`ceres::CostFunction`中的parameter_block_sizes_
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    // n为要保留下来的变量个数
    set_num_residuals(marginalization_info->n);
};

// 思路: 把边缘化的先验信息变成约束项
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    // n 要保留的变量 总维度  n = 相邻帧的帧的IMU因子(速度,ba,bg) + 10帧位姿 + 外参 = 1x9 + 10x6 +1x6 = 75
    int n = marginalization_info->n;
    // m 要marg的变量总维度  m = 要marg的IMU因子(速度,ba,bg) + marg的帧的位姿 + marg的帧的路标信息 =
    int m = marginalization_info->m;

    /// 计算当前相对于线性化点的边缘化先验残差
    Eigen::VectorXd dx(n);
    // 遍历， 遍历 "要保留的变量的数目" 次
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        // size ： 要保留的第i个变量的GLOBAL维度数
        int size = marginalization_info->keep_block_size[i];
        // marginalization_info->keep_block_idx[i] : 要保留的第i个变量的 local size idx 减去 要marg掉的变量总维度(local)
        // 相当于，得到相对于 要保留的第0个变量的 local size 的idx
        int idx = marginalization_info->keep_block_idx[i] - m;
        // 待优化参数: 根据上一次marg的时候，得到的要保留的变量的地址
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        // 要保留的第i个变量的 数据，维度为size
        // keep_block_data[]: 要保留的变量i的地址?
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        // 变量的global size不等于7，那就是IMU的因子
        if (size != 7)
            // ??
            dx.segment(idx, size) = x - x0;
        else
        {
            // 位姿因子
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // n 要保留的变量 总维度  n = 相邻帧的帧的IMU因子(速度,ba,bg) + 10帧位姿 + 外参 = 1x9 + 10x6 +1x6 = 75
    // 基于线性化点，更新边缘化残差
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {
        // 遍历要保留的参数
        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            // 计算边缘化残差相对于第i个参数的雅克比
            if (jacobians[i])
            {
                // size: 要保留的变量的 global size
                // local_size: 要保留的变量的 local size
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                // idx: 相对于 要保留的第0个变量的 local size 的idx
                int idx = marginalization_info->keep_block_idx[i] - m;
                // 指针构造Eigen, 取对应该变量的雅克比部分
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                // jacobian: n x size
                jacobian.setZero();
                ///                                                                         linearized_jacobians:
                /// dr[0]/d par ---   | dr[0]/d par_0[0] , dr[0]/ d par_0[1] , dr[0]/ d par_0[2] ... | dr[0]/d par_1[0] , dr[0]/ d par_1[1] , dr[0]/ d par_1[2] ... |                          ...                                 | dr[0]/d par_i[0] , dr[0]/ d par_i[1] , dr[0]/ d par_i[2] ... |
                /// dr[1]/d par ---   | dr[1]/d par_0[0] , dr[1]/ d par_0[1] , dr[1]/ d par_0[2] ... | dr[1]/d par_1[0] , dr[1]/ d par_1[1] , dr[1]/ d par_1[2] ... |                          ...                                 | dr[1]/d par_i[0] , dr[1]/ d par_i[1] , dr[1]/ d par_i[2] ... |
                /// ...               |                          ...                                 |                          ...                                 |                          ...                                 |                          ...                                 |
                /// dr[n]/d par ---   | dr[n]/d par_0[0] , dr[n]/ d par_0[1] , dr[n]/ d par_0[2] ... | dr[n]/d par_1[0] , dr[n]/ d par_1[1] , dr[n]/ d par_1[2] ... |                          ...                                 | dr[n]/d par_i[0] , dr[n]/ d par_i[1] , dr[n]/ d par_i[2] ... |
                ///
                // 雅克比直接取之前计算的线性化点处的雅克比，取矩阵的中间对应的列
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
