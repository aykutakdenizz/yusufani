#include <stdio.h> 
#include <stdlib.h>
void BubbleSort(int * arr, int n);
void SelectionSort(int * arr, int n);
void InsertionSort(int * arr, int n);
void ShellSort(int* arr, int n);

int main() {
	int *arr, n, i;
	printf("Lutfen dizi boyutunu giriniz :");
	scanf("%d", &n);
	arr = (int *)calloc(n, sizeof(int));
	for (i = 0; i < n; i++) {
		printf("Dizinin %d. elemanini giriniz", i + 1);
		scanf("%d", &arr[i]);
	}
	printf("Lutfen Siralama Metodunu Seciniz:\n1-Bubble Sort\n2-Selection Sort\n3-Insertion Sort\n4-Shell Sort");
	scanf("%d", &i);
	switch (i) {
		case 1: {
			BubbleSort(arr, n);
			break;
		}
		case 2: {
			SelectionSort(arr, n);
			break;
		}
		case 3: {
			InsertionSort(arr, n);
			break;
		}
		case 4: {
			ShellSort(arr, n);
			break;
		}
	}
	printf("Siralanmis Dizi\n");
	for (i = 0; i < n; i++) {
		printf("%d\t", arr[i]);
	}
	scanf("%d");
}
void BubbleSort(int * arr, int n) {
	int i = 0, j, xchg = 1, tmp;
	while (i < n - 1 && xchg) {
		xchg = 0;
		for (j = 0; j < n - 1 - i; j++) {
			if (arr[j] > arr[j + 1]) {
				tmp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = tmp;
				xchg = 1;
			}
		}
		i++;
	}

}
void SelectionSort(int * arr, int n) {
	int i, j, min, yer;
	for (i = 0; i < n - 1; i++) {
		min = arr[i];
		yer = i;
			for (j = i + 1; j < n; j++) {
				if (arr[j] < min) {
					min = arr[j];
					yer = j;
				}
			}
		arr[yer] = arr[i];
		arr[i] = min;
	}
}
void InsertionSort(int * arr, int n) {
	int i, j, min;
	for (i = 1; i < n; i++) {
		min = arr[i];
		j = i - 1;
		while (j >= 0 && min < arr[j]) {
			arr[j + 1] = arr[j];
			j--;
		}
		arr[j + 1] = min;
	}
}

void ShellSort(int* arr, int n)
{
	for (int k = n / 2; k > 0; k /= 2)
	{
		for (int i = k; i < n; i++)
		{
			int temp = arr[i];
			int j = i;
			while (j >= k && arr[j - k] > temp)
			{
				arr[j] = arr[j - k];
				j -= k;
			}
			arr[j] = temp;
		}
	}
}
