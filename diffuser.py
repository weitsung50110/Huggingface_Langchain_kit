from diffusers import StableDiffusionPipeline
import torch
import argparse


# 定義 main 函數，接收一個參數 output_filename
def main(prompt, output_filename):
    # 設定模型 ID
    model_id = "runwayml/stable-diffusion-v1-5"

    # 從預訓練模型載入 Stable Diffusion 管道
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")  # 使用 CPU 運行

    # 定義生成圖像的提示詞
    # prompt = "taiwanese handsome boy"

    # 生成圖像
    image = pipe(prompt).images[0]

    # 將圖像保存到指定的文件名
    image.save(output_filename)

# 檢查程式是否以主程式運行
if __name__ == "__main__":
    # 創建參數解析器
    parser = argparse.ArgumentParser(description="Generate an image with Stable Diffusion and save it.")

    # 添加 --prompt 參數，用於指定生成圖像的提示詞
    parser.add_argument("--prompt", type=str, required=True, help="The prompt for generating the image.")
    # 添加 --output 參數，用於指定輸出文件名
    parser.add_argument("--output", type=str, required=True, help="The output filename for the generated image.")

    # 解析命令列參數
    args = parser.parse_args()

    # 呼叫 main 函數並傳遞解析到的輸出文件名
    main(args.prompt, args.output)
