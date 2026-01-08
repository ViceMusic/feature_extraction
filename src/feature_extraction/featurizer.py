"""
肽类分子特征提取器模块

提供 PeptideFeaturizer 类，用于从 SMILES 字符串提取分子特征，
包括 Morgan 指纹、Avalon 指纹、QED 属性和物理化学描述符。
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem, QED, rdMolDescriptors, Crippen, Descriptors
)

import torch
from transformers import AutoTokenizer, AutoModel

# 尝试导入 Avalon 指纹支持（可选）
try:
    from rdkit.Avalon import pyAvalonTools
    _HAS_AVALON = True
except (ImportError, ModuleNotFoundError):
    pyAvalonTools = None
    _HAS_AVALON = False

# 检查是否支持transformer和ChemBERTa
# 尝试检查环境是否支持 Transformers
try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    _HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class PeptideFeaturizer:
    """
    肽类分子特征提取器。

    用于从 SMILES 字符串提取分子特征，包括：
    - QED 属性（分子量、LogP、HBA、HBD、PSA、旋转键数、芳香性、警报）
    - 物理化学描述符（脂溶性、刚性、分子大小等）
    - Gasteiger 电荷统计
    - Morgan 指纹（位向量）
    - Avalon 指纹（可选）

    Attributes:
        morgan_bits (int): Morgan 指纹位数（默认 1024）
        avalon_bits (int): Avalon 指纹位数（默认 512）
        use_avalon (bool): 是否使用 Avalon 指纹（默认 True）
    """

    def __init__(
        self,
        morgan_bits: int = 1024,
        avalon_bits: int = 512,
        chemberta_bits: int=384,
        use_avalon: bool = True,
        use_chemberta: bool = False, # 新增：默认关闭，因为加载模型较慢
        model_name: str = "deepchem/ChemBERTa-77M-MLM" # 新增：模型路径
    ) -> None:
        """
        初始化肽类特征提取器。

        Args:
            morgan_bits (int): Morgan 指纹位数。默认 1024。
            avalon_bits (int): Avalon 指纹位数。默认 512。
            use_avalon (bool): 是否使用 Avalon 指纹。默认 True。
                如果系统不支持 Avalon，将自动设置为 False。
        """
        self.morgan_bits = morgan_bits
        self.avalon_bits = avalon_bits
        self.use_avalon = use_avalon and _HAS_AVALON
        
        if use_avalon and not _HAS_AVALON:
            logger.warning(
                "Avalon 指纹在此环境中不可用，已禁用。"
                "若需使用，请确保 RDKit 编译时启用了 Avalon 支持。"
            )

        # --- 新增 ChemBERTa 初始化 ---
        self.use_chemberta = use_chemberta and _HAS_TRANSFORMERS
        if self.use_chemberta:
            logger.info(f"正在加载 ChemBERTa 模型: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval() # 设置为评估模式
            # 如果有 GPU，可以移动到 GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        # ---------------------------

    @staticmethod
    def _safe_float(value: any, default: float = 0.0) -> float:
        """
        安全地将值转换为浮点数，处理 NaN 和 Inf。

        Args:
            value: 待转换的值。
            default (float): 转换失败时的默认值。

        Returns:
            float: 转换后的浮点数。
        """
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return default
            return val
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _gasteiger_charge_stats(mol: Chem.Mol) -> List[float]:
        """
        计算分子的 Gasteiger 电荷统计。

        Args:
            mol (Chem.Mol): RDKit 分子对象。

        Returns:
            List[float]: 包含 5 个统计值的列表
                [均值, 最大值, 最小值, 标准差, 总和]。
                若计算失败，返回全零列表。
        """
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for i in range(mol.GetNumAtoms()):
                try:
                    charge = float(
                        mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge")
                    )
                    charges.append(charge)
                except (TypeError, KeyError):
                    charges.append(0.0)
            
            if len(charges) == 0:
                charges = [0.0]
            
            charges_arr = np.asarray(charges, dtype=np.float64)
            # 替换非有限值为 0
            charges_arr = np.where(np.isfinite(charges_arr), charges_arr, 0.0)
            
            return [
                float(np.mean(charges_arr)),
                float(np.max(charges_arr)),
                float(np.min(charges_arr)),
                float(np.std(charges_arr)),
                float(np.sum(charges_arr)),
            ]
        except Exception as e:
            logger.debug(f"计算 Gasteiger 电荷时出错: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _qed_properties(mol: Chem.Mol) -> List[float]:
        """
        提取 QED（药物相似性）属性。

        Args:
            mol (Chem.Mol): RDKit 分子对象。

        Returns:
            List[float]: 包含 8 个 QED 属性的列表
                [MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS]。
                若计算失败，返回全零列表。
        """
        try:
            props = QED.properties(mol)
            return [
                float(props.MW),
                float(props.ALOGP),
                float(props.HBA),
                float(props.HBD),
                float(props.PSA),
                float(props.ROTB),
                float(props.AROM),
                float(props.ALERTS),
            ]
        except Exception as e:
            logger.debug(f"计算 QED 属性时出错: {e}")
            return [0.0] * 8

    @staticmethod
    def _physchem_descriptors(mol: Chem.Mol) -> List[float]:
        """
        提取物理化学描述符（脂溶性、刚性、分子大小等）。

        Args:
            mol (Chem.Mol): RDKit 分子对象。

        Returns:
            List[float]: 包含 11 个物理化学描述符的列表
                [分子量, LogP, HBA, HBD, TPSA, 旋转键数, 环数, Fsp3, 
                 重原子数, 原子总数, 刚性指标]。
                若计算失败，返回全零列表。
        """
        try:
            mw = float(Descriptors.MolWt(mol))
            logp = float(Crippen.MolLogP(mol))
            hba = float(rdMolDescriptors.CalcNumHBA(mol))
            hbd = float(rdMolDescriptors.CalcNumHBD(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            rotb = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
            num_rings = float(rdMolDescriptors.CalcNumRings(mol))
            fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
            heavy_atoms = float(Descriptors.HeavyAtomCount(mol))
            total_atoms = float(mol.GetNumAtoms())
            
            # 简单的刚性指标：环数 / (1 + 旋转键数)
            rigidity_proxy = num_rings / (1.0 + rotb)
            
            return [
                mw, logp, hba, hbd, tpsa, rotb, num_rings, fsp3,
                heavy_atoms, total_atoms, rigidity_proxy
            ]
        except Exception as e:
            logger.debug(f"计算物理化学描述符时出错: {e}")
            return [0.0] * 11

    def _morgan_fingerprint(
        self, mol: Chem.Mol, radius: int = 2, use_chirality: bool = True
    ) -> List[float]:
        """
        提取 Morgan 指纹（位向量）。

        Args:
            mol (Chem.Mol): RDKit 分子对象。
            radius (int): Morgan 指纹半径。默认 2。
            use_chirality (bool): 是否考虑手性。默认 True。

        Returns:
            List[float]: Morgan 指纹位向量，长度为 morgan_bits。
                若计算失败，返回全零列表。
        """
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=self.morgan_bits, useChirality=use_chirality
            )
            arr = np.zeros(self.morgan_bits, dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr.astype(np.float32).tolist()
        except Exception as e:
            logger.debug(f"计算 Morgan 指纹时出错: {e}")
            return [0.0] * self.morgan_bits

    def _avalon_fingerprint(self, mol: Chem.Mol) -> Optional[List[float]]:
        """
        提取 Avalon 指纹（位向量）。

        Args:
            mol (Chem.Mol): RDKit 分子对象。

        Returns:
            Optional[List[float]]: Avalon 指纹位向量，长度为 avalon_bits。
                若不支持或计算失败，返回 None。
        """
        if not self.use_avalon:
            return None
        
        try:
            fp = pyAvalonTools.GetAvalonFP(mol, self.avalon_bits)
            arr = np.zeros(self.avalon_bits, dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr.astype(np.float32).tolist()
        except Exception as e:
            logger.debug(f"计算 Avalon 指纹时出错: {e}")
            return [0.0] * self.avalon_bits

    # --- 新增：ChemBERTa 特征提取私有方法(但是这里面完全没看到ChemBERTa相关的库啊) ---
    def _chemberta_fingerprint(self, smiles: str) -> Optional[List[float]]:
        if not self.use_chemberta:
            return None
        try:
            inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取最后一层隐藏层所有 token 的平均值 (Mean Pooling)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
                
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.debug(f"计算 ChemBERTa 特征时出错: {e}")
            # 注意：ChemBERTa-77M 的维度通常是 384
            return [0.0] * self.model.config.hidden_size

    def featurize(
        self, smiles: str
    ) -> Tuple[Optional[List[float]], bool]:
        """
        从 SMILES 字符串提取分子特征。

        Args:
            smiles (str): 分子的 SMILES 字符串。

        Returns:
            Tuple[Optional[List[float]], bool]: 
                - 第一个元素：特征列表，若提取失败则为 None
                - 第二个元素：是否成功提取（bool）
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.debug(f"无效的 SMILES 字符串: {smiles}")
                return None, False
            
            # 依次提取各部分特征
            qed_feats = self._qed_properties(mol)
            physchem_feats = self._physchem_descriptors(mol)
            charge_feats = self._gasteiger_charge_stats(mol)
            morgan_feats = self._morgan_fingerprint(mol)
            
            # 组合所有特征
            all_features = (
                qed_feats + physchem_feats + charge_feats + morgan_feats
            )
            
            # 如果启用了 Avalon，添加 Avalon 指纹
            if self.use_avalon:
                avalon_feats = self._avalon_fingerprint(mol)
                if avalon_feats is not None:
                    all_features.extend(avalon_feats)
            
            # --- 新增：整合 ChemBERTa ---
            if self.use_chemberta:
                berta_feats = self._chemberta_fingerprint(smiles)
                if berta_feats is not None: 
                    all_features.extend(berta_feats)
            
            return all_features, True
        
        except Exception as e:
            logger.error(f"提取 SMILES '{smiles}' 的特征时出错: {e}")
            return None, False

    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表。

        返回的特征顺序与 featurize() 方法返回的特征列表一致。

        Returns:
            List[str]: 特征名称列表。
        """
        names = []
        
        # QED 属性名
        names.extend([
            "QED_MW", "QED_ALOGP", "QED_HBA", "QED_HBD",
            "QED_PSA", "QED_ROTB", "QED_AROM", "QED_ALERTS"
        ])
        
        # 物理化学描述符名
        names.extend([
            "PC_MolWt", "PC_LogP", "PC_HBA", "PC_HBD", "PC_TPSA",
            "PC_RotB", "PC_Rings", "PC_FractionCSP3", "PC_HeavyAtomCount",
            "PC_NumAtoms", "PC_RigidityProxy"
        ])
        
        # Gasteiger 电荷统计名
        names.extend([
            "GC_Mean", "GC_Max", "GC_Min", "GC_Std", "GC_Sum"
        ])
        
        # Morgan 指纹名
        names.extend([f"Morgan_{i}" for i in range(self.morgan_bits)])
        
        # Avalon 指纹名（如果启用）
        if self.use_avalon:
            names.extend([f"Avalon_{i}" for i in range(self.avalon_bits)])

        # --- 新增：ChemBERTa 维度名称 ---
        if self.use_chemberta:
            hidden_size = self.model.config.hidden_size
            names.extend([f"ChemBERTa_{i}" for i in range(hidden_size)])
        
        return names

    @property
    def n_features(self) -> int:
        """
        获取特征总数。

        Returns:
            int: 特征总数。
        """
        n_base = 8 + 11 + 5 + self.morgan_bits
        if self.use_avalon:
            n_base += self.avalon_bits
        return n_base

