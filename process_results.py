import os
import shutil
import pandas as pd

receptors = [
    "brain_cancer.pdb",
    "brain_cancer_2.pdb",
    "normal_brain_cell.pdb",
    "blood_exmpl.pdb"
]

docking_dir = "Docking_result"

def main():
    os.makedirs(docking_dir, exist_ok=True)
    raw_dir = os.path.join(docking_dir, "raw_data")
    critical_dir = os.path.join(docking_dir, "critical_data")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(critical_dir, exist_ok=True)
    
    # 1) Eski tabloları (MD ve CSV) sil
    for old_file in ["docking_summary.md", "docking_summary.csv"]:
        old_path = os.path.join(critical_dir, old_file)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
                print(f"Eski dosya silindi: {old_file}")
            except Exception as e:
                print(f"Silinemedi: {old_file} - {e}")
                
    excel_path = os.path.join(critical_dir, "Modern_Docking_Results.xlsx")
    
    # 2) Pandas ile modern Excel tablosunu oluştur
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for receptor in receptors:
            base_name = os.path.splitext(receptor)[0]
            out_file = f"{base_name}_result.out"
            
            # Geri dönüşüm (daha önce raw_data içine taşınmışsa oradan oku)
            if not os.path.exists(out_file) and os.path.exists(os.path.join(raw_dir, out_file)):
                out_path = os.path.join(raw_dir, out_file)
                needs_move = False
            else:
                out_path = out_file
                needs_move = True
                
            scores = []
            rmsds = []
            
            if os.path.exists(out_path):
                with open(out_path, 'r', encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                
                count = 0
                for line in lines:
                    parts = line.split()
                    if len(parts) == 9:
                        try:
                            floats = [float(p) for p in parts]
                            count += 1
                            scores.append(floats[6])
                            rmsds.append(floats[7])
                            if count >= 5:
                                break
                        except ValueError:
                            continue
                            
                # Dosyayı raw_data klasörüne taşı (eğer ana dizindeyse)
                if needs_move:
                    shutil.move(out_path, os.path.join(raw_dir, out_file))
            else:
                print(f"Uyarı: {out_file} bulunamadı.")
                
            # 5 sonuca tamamla
            while len(scores) < 5:
                scores.append(None)
                rmsds.append(None)
                
            # Yatay modernize DataFrame oluştur
            df = pd.DataFrame({
                "Rank": [1, 2, 3, 4, 5],
                "Docking Score": scores,
                "Ligand RMSD (Å)": rmsds
            })
            
            # Transpoze et (Sütunlar 1, 2, 3, 4, 5 olsun)
            df = df.set_index("Rank").T
            df.columns = [1, 2, 3, 4, 5]
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Metric"}, inplace=True)
            
            # Excel sekmesine (sheet) yaz (Sheet isim limiti: max 31 karakter)
            sheet_name = base_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Openpyxl ile modern tasarım ayarlamaları
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Sütun genişlikleri
            worksheet.column_dimensions['A'].width = 20
            for col_letter in ['B', 'C', 'D', 'E', 'F']:
                worksheet.column_dimensions[col_letter].width = 12
                
    print(f"\n[BAŞARILI] Modern Excel dosyası başarıyla oluşturuldu: {excel_path}")
    print(f"[BİLGİ] İşlenmemiş ham (.out) veriler klasöre taşındı: {raw_dir}")

if __name__ == "__main__":
    main()
